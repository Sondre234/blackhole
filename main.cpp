// main.cpp — Schwarzschild lensing + accretion disk + free-fly camera (WASD + mouse)
// Requires OpenGL 4.3+ (compute shaders), GLFW, GLAD.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

static void die(const char* msg) {
  std::cerr << msg << "\n";
  std::exit(1);
}

static GLuint compileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    GLint len = 0;
    glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
    std::string log(len, '\0');
    glGetShaderInfoLog(s, len, nullptr, log.data());
    std::cerr << "Shader compile error:\n" << log << "\n";
    std::exit(1);
  }
  return s;
}

static GLuint linkProgram(const std::vector<GLuint>& shaders) {
  GLuint p = glCreateProgram();
  for (auto s : shaders) glAttachShader(p, s);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    GLint len = 0;
    glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
    std::string log(len, '\0');
    glGetProgramInfoLog(p, len, nullptr, log.data());
    std::cerr << "Program link error:\n" << log << "\n";
    std::exit(1);
  }
  for (auto s : shaders) {
    glDetachShader(p, s);
    glDeleteShader(s);
  }
  return p;
}

// ------------------------- Free-fly camera -------------------------
static float gCamX = 0.0f, gCamY = 0.35f, gCamZ = 20.0f;
static float gYaw = -90.0f, gPitch = 0.0f;
static float gLastX = 0.0f, gLastY = 0.0f;
static bool  gFirstMouse = true;
static float gMouseSens = 0.08f;
static float gMoveSpeed = 5.0f; // units/sec (scaled by dt)

static void mouseCallback(GLFWwindow*, double xpos, double ypos) {
  if (gFirstMouse) {
    gLastX = (float)xpos;
    gLastY = (float)ypos;
    gFirstMouse = false;
  }
  float xoff = (float)xpos - gLastX;
  float yoff = gLastY - (float)ypos; // inverted
  gLastX = (float)xpos;
  gLastY = (float)ypos;

  xoff *= gMouseSens;
  yoff *= gMouseSens;

  gYaw   += xoff;
  gPitch += yoff;

  if (gPitch > 89.0f) gPitch = 89.0f;
  if (gPitch < -89.0f) gPitch = -89.0f;
}

static void computeCameraBasis(float basis9[9], float pos3[3]) {
  const float deg2rad = 0.01745329252f;
  float yawR = gYaw * deg2rad;
  float pitR = gPitch * deg2rad;

  float fx = std::cos(yawR) * std::cos(pitR);
  float fy = std::sin(pitR);
  float fz = std::sin(yawR) * std::cos(pitR);

  float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
  fx/=fl; fy/=fl; fz/=fl;

  // world up
  float ux = 0.0f, uy = 1.0f, uz = 0.0f;

  // right = normalize(forward x up)
  float rx = fy*uz - fz*uy;
  float ry = fz*ux - fx*uz;
  float rz = fx*uy - fy*ux;
  float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
  rx/=rl; ry/=rl; rz/=rl;

  // up = right x forward
  float upx = ry*fz - rz*fy;
  float upy = rz*fx - rx*fz;
  float upz = rx*fy - ry*fx;

  pos3[0] = gCamX; pos3[1] = gCamY; pos3[2] = gCamZ;

  // mat3 columns: right, up, forward (matches shader)
  basis9[0]=rx;  basis9[1]=ry;  basis9[2]=rz;
  basis9[3]=upx; basis9[4]=upy; basis9[5]=upz;
  basis9[6]=fx;  basis9[7]=fy;  basis9[8]=fz;
}

static const char* kComputeSrc = R"(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D img;

uniform ivec2 uResolution;

// Camera
uniform vec3  uCamPos;
uniform mat3  uCamBasis;   // columns: right, up, forward
uniform float uFovY;

// Black hole
uniform float uM;           // geometric units (G=c=1), rs=2M

// Ray integration
uniform float uPhiStep;
uniform int   uMaxSteps;
uniform float uFarR;

// Accretion disk (world plane y=0)
uniform float uDiskInnerR;     // e.g. 3.2*M
uniform float uDiskOuterR;     // e.g. 22*M
uniform float uDiskHalfThick;  // e.g. 0.03*M
uniform float uDiskBoost;      // e.g. 2.5
uniform int   uSkyMode;        // 0=stars, 1=grid
uniform float uTime;
uniform float uPlanetOrbitR;
uniform float uPlanetRadius;
uniform float uPlanetOmega;
uniform float uPlanetY;
uniform float uCollapse;
uniform float uStarRadiusStart;
uniform float uStarRadiusEnd;

const float PI = 3.141592653589793;

float clamp01(float x) { return max(0.0, min(1.0, x)); } // avoids clamp() overload ambiguity

vec3 skyColor(vec3 dir, float lensStrength) {
  dir = normalize(dir);
  lensStrength = clamp01(lensStrength);

  if (uSkyMode == 1) {
    float lon = atan(dir.z, dir.x);                  // -pi..pi
    float lat = asin(max(-1.0, min(1.0, dir.y)));    // -pi/2..pi/2
    vec2 uv = vec2(lon/(2.0*PI)+0.5, lat/PI+0.5);

    // Stronger apparent grid distortion for rays that passed closer to the horizon.
    float stretchX = 1.0 + 2.8 * lensStrength * lensStrength;
    float stretchY = 1.0 + 1.2 * lensStrength;
    uv = (uv - 0.5) * vec2(stretchX, stretchY) + 0.5;

    vec2 gridScale = vec2(24.0, 12.0) * (1.0 + 2.2 * lensStrength);
    float gx = abs(fract(uv.x * gridScale.x) - 0.5);
    float gy = abs(fract(uv.y * gridScale.y) - 0.5);

    float lineW = mix(0.02, 0.006, lensStrength);
    float line = smoothstep(lineW, 0.0, min(gx, gy));
    vec3 bg = vec3(0.02);
    vec3 fg = vec3(0.9, 0.92, 0.96) * (1.0 + 0.35 * lensStrength);
    return bg + fg * line;
  }

  float t = clamp01(dir.y*0.5 + 0.5);
  vec3 col = mix(vec3(0.02,0.02,0.03), vec3(0.10,0.14,0.25), t);

  float h = fract(sin(dot(dir*437.0, vec3(12.9898,78.233,37.719))) * 43758.5453);
  float star = smoothstep(0.9985, 1.0, h);
  col += star * vec3(2.0);
  return col;
}

// Wrapper to preserve old call sites (e.g. skyColor(v))
vec3 skyColor(vec3 dir) {
  return skyColor(dir, 0.0);
}

bool segmentHitsDisk(vec3 p0, vec3 p1, out vec3 hitPos, out float hitT) {
  // Intersect segment with plane y=0
  float y0 = p0.y, y1 = p1.y;
  if ((y0 > 0.0 && y1 > 0.0) || (y0 < 0.0 && y1 < 0.0)) return false;
  float denom = (y0 - y1);
  if (abs(denom) < 1e-8) return false;

  float t = y0 / denom; // 0..1
  if (t < 0.0 || t > 1.0) return false;

  vec3 p = mix(p0, p1, t);

  // Thickness band around y=0
  if (abs(p.y) > uDiskHalfThick) return false;

  float r = length(p.xz);
  if (r < uDiskInnerR || r > uDiskOuterR) return false;

  hitPos = p;
  hitT = t;
  return true;
}

bool segmentHitsSphere(vec3 p0, vec3 p1, vec3 center, float radius, out vec3 hitPos, out float hitT) {
  vec3 d = p1 - p0;
  vec3 oc = p0 - center;
  float a = dot(d, d);
  float b = 2.0 * dot(oc, d);
  float c = dot(oc, oc) - radius * radius;
  float disc = b*b - 4.0*a*c;
  if (disc < 0.0 || a < 1e-8) return false;

  float root = sqrt(disc);
  float t0 = (-b - root) / (2.0 * a);
  float t1 = (-b + root) / (2.0 * a);
  float t = 1e9;
  if (t0 >= 0.0 && t0 <= 1.0) t = t0;
  if (t1 >= 0.0 && t1 <= 1.0) t = min(t, t1);
  if (t > 1.0) return false;

  hitT = t;
  hitPos = p0 + d * t;
  return true;
}

vec3 shadePlanet(vec3 p, vec3 center) {
  vec3 n = normalize(p - center);
  vec3 sunDir = normalize(vec3(0.8, 0.35, -0.45));
  float lit = max(dot(n, sunDir), 0.0);

  float bands = 0.85 + 0.15 * sin((p.y - center.y) * 15.0 + uTime * 0.6);
  vec3 deep = vec3(0.02, 0.10, 0.35);
  vec3 bright = vec3(0.30, 0.65, 1.00);
  vec3 base = mix(deep, bright, lit * 0.85 + 0.15);
  return base * bands + vec3(0.02, 0.05, 0.10) * pow(lit, 8.0);
}

vec3 shadeHypergiant(vec3 p, vec3 center) {
  vec3 n = normalize(p - center);
  float turbulent = 0.55 + 0.45 * sin(n.x * 15.0 + uTime * 0.9)
                    + 0.20 * sin(n.z * 25.0 - uTime * 1.6)
                    + 0.10 * sin((n.x + n.y) * 70.0 + uTime * 4.0);
  turbulent = clamp01(0.5 + 0.5 * turbulent);

  vec3 deep = vec3(0.38, 0.03, 0.01);
  vec3 bright = vec3(1.35, 0.20, 0.03);
  vec3 col = mix(deep, bright, turbulent);

  float simmer = 0.82 + 0.18 * sin(uTime * 2.8 + n.y * 20.0);
  float collapseGlow = 1.0 + 0.7 * pow(clamp01(uCollapse), 3.0);
  return col * simmer * collapseGlow;
}

vec3 shadeSupernova(vec3 p, vec3 center, float phase) {
  vec3 n = normalize(p - center);
  float flicker = 0.85 + 0.15 * sin(dot(n, vec3(13.0, 21.0, 8.0)) + uTime * 16.0);
  vec3 core = vec3(7.0, 3.0, 0.8);
  vec3 edge = vec3(1.5, 0.45, 0.1);
  vec3 col = mix(edge, core, pow(phase, 0.6));
  return col * flicker;
}

vec3 shadeDisk(vec3 p, vec3 segDir, float M) {
  float r = length(p.xz);
  float azimuth = atan(p.z, p.x);

  float t = (r - uDiskInnerR) / max(uDiskOuterR - uDiskInnerR, 1e-6);
  float x = clamp01(t);
  float hot = pow(1.0 - x, 1.8);

  // HDR-ish warm gradient
  vec3 cool = vec3(0.15, 0.03, 0.01);
  vec3 warm = vec3(6.0,  2.2,  0.35);
  vec3 col  = mix(cool, warm, hot);

  // Advected turbulence to keep the disk visually alive even when the camera is still.
  float spinRate = 0.7 / max(sqrt(r), 0.25);
  float swirl = sin(azimuth * 20.0 - uTime * 7.0 * spinRate + r * 2.3);
  float bands = sin(azimuth * 57.0 - uTime * 16.0 * spinRate - r * 7.0);
  float turbulence = 0.75 + 0.25 * swirl + 0.12 * bands;
  col *= max(turbulence, 0.4);

  // Disk rotates around Y: tangent direction
  vec3 tangent = normalize(vec3(-p.z, 0.0, p.x));

  // "Kepler-ish" speed, clamped
  float beta  = clamp01(sqrt(M / max(r, 1e-4)) * 0.68) * 0.82; // up to ~0.82
  float gamma = inversesqrt(1.0 - beta*beta);

  // segDir points along the ray as we step outward; direction to camera is -segDir
  float mu  = dot(tangent, normalize(-segDir));
  float dop = 1.0 / (gamma * (1.0 - beta * mu));
  float beam = pow(max(dop, 0.35), 3.5);

  // mild dimming near horizon (artistic)
  float rs = 2.0 * M;
  float g  = sqrt(max(1.0 - rs / max(r, rs*1.001), 0.0));
  col *= (0.25 + 0.75 * g);

  col *= uDiskBoost * beam;
  return col;
}

// Integrate u'' + u = 3 M u^2, where u=1/r, parameterized by phi (equatorial-plane reduction).
vec4 traceSchwarzschild(vec3 camPos, vec3 rayDir) {
  float collapse = clamp01(uCollapse);

  vec3 starCenter = vec3(0.0);
  float starRadius = mix(uStarRadiusStart, uStarRadiusEnd, collapse);
  vec3 starRayEnd = camPos + rayDir * uFarR;

  // Phase 1: red hypergiant (no black hole, no relativistic lensing yet).
  if (collapse < 0.72) {
    vec3 starHitPos;
    float starHitT;
    bool hasStarHit = segmentHitsSphere(camPos, starRayEnd, starCenter, starRadius, starHitPos, starHitT);
    if (hasStarHit) return vec4(shadeHypergiant(starHitPos, starCenter), 1.0);
    return vec4(skyColor(rayDir, 0.0), 1.0);
  }

  // Phase 2: supernova flash (still no black-hole lensing).
  if (collapse < 0.84) {
    float snPhase = smoothstep(0.72, 0.84, collapse);
    float shellRadius = mix(starRadius, uStarRadiusStart * 1.35, snPhase);
    float shellThickness = mix(0.65 * uM, 0.35 * uM, snPhase);

    vec3 hitOuterPos, hitInnerPos;
    float tOuter, tInner;
    bool hitOuter = segmentHitsSphere(camPos, starRayEnd, starCenter, shellRadius, hitOuterPos, tOuter);
    bool hitInner = segmentHitsSphere(camPos, starRayEnd, starCenter, max(shellRadius - shellThickness, 0.01), hitInnerPos, tInner);
    if (hitOuter && (!hitInner || tOuter < tInner)) {
      return vec4(shadeSupernova(hitOuterPos, starCenter, snPhase), 1.0);
    }
    return vec4(skyColor(rayDir, 0.0), 1.0);
  }

  float bhPhase = smoothstep(0.84, 1.0, collapse);
  float M  = mix(0.22 * uM, uM, bhPhase);
  float rs = 2.0 * M;

  vec3 r0 = camPos;
  float rObs = length(r0);
  if (rObs <= rs * 1.001) return vec4(0,0,0,1);

  vec3 a = normalize(r0);           // radial at observer
  vec3 v = normalize(rayDir);

  // Ray plane normal
  vec3 n = cross(r0, v);
  float nlen = length(n);
  if (nlen < 1e-6) return vec4(skyColor(v), 1.0);
  n /= nlen;

  // Tangential axis in the plane (direction of increasing phi)
  vec3 b = normalize(cross(n, a));

  float v_r = dot(v, a);
  float v_t = dot(v, b);
  if (v_t < 0.0) { b = -b; v_t = -v_t; }

  float L = max(rObs * v_t, 1e-6);

  float u = 1.0 / rObs;
  float p = -(v_r) / L;     // p = du/dphi
  float phi = 0.0;

  float rPrev = rObs;
  float minR = rObs;
  vec3 posPrev = (a*cos(phi) + b*sin(phi)) * rPrev;

  for (int i = 0; i < uMaxSteps; i++) {
    float r = 1.0 / u;
    if (r <= rs * 1.001) return vec4(0,0,0,1);

    // RK4 in phi: u' = p ; p' = 3 M u^2 - u
    float h = uPhiStep;

    float k1_u = p;
    float k1_p = 3.0*M*u*u - u;

    float u2 = u + 0.5*h*k1_u;
    float p2 = p + 0.5*h*k1_p;
    float k2_u = p2;
    float k2_p = 3.0*M*u2*u2 - u2;

    float u3 = u + 0.5*h*k2_u;
    float p3 = p + 0.5*h*k2_p;
    float k3_u = p3;
    float k3_p = 3.0*M*u3*u3 - u3;

    float u4 = u + h*k3_u;
    float p4 = p + h*k3_p;
    float k4_u = p4;
    float k4_p = 3.0*M*u4*u4 - u4;

    u   += (h/6.0) * (k1_u + 2.0*k2_u + 2.0*k3_u + k4_u);
    p   += (h/6.0) * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p);
    phi += h;

    if (u <= 0.0) {
      float lensStrength = clamp01((12.0 * M - minR) / (10.0 * M)) * bhPhase;
      return vec4(skyColor(v, lensStrength), 1.0);
    }

    float rNow = 1.0 / u;
    minR = min(minR, rNow);
    vec3 posNow = (a*cos(phi) + b*sin(phi)) * rNow;

    vec3 planetCenter = vec3(
      cos(uTime * uPlanetOmega) * uPlanetOrbitR,
      uPlanetY,
      sin(uTime * uPlanetOmega) * uPlanetOrbitR
    );

    vec3 hitDiskPos, hitPlanetPos;
    float tDisk = 2.0;
    float tPlanet = 2.0;
    bool hasDisk = segmentHitsDisk(posPrev, posNow, hitDiskPos, tDisk) && bhPhase > 0.15;
    bool hasPlanet = segmentHitsSphere(posPrev, posNow, planetCenter, uPlanetRadius, hitPlanetPos, tPlanet);

    if (hasPlanet && (!hasDisk || tPlanet < tDisk)) {
      return vec4(shadePlanet(hitPlanetPos, planetCenter), 1.0);
    }

    if (hasDisk) {
      vec3 segDir = normalize(posNow - posPrev);
      vec3 c = shadeDisk(hitDiskPos, segDir, M) * smoothstep(0.15, 0.90, bhPhase);
      return vec4(c, 1.0);
    }
    posPrev = posNow;

    // Escape to far sphere while moving outward: dr/dlambda = -L*p
    float dr_dlambda = -L * p;
    if (rNow >= uFarR && dr_dlambda > 0.0) {
      // approximate outgoing direction using plane basis derivative
      float dr_dphi = -p / (u*u);
      float c = cos(phi);
      float s = sin(phi);
      float dx = dr_dphi * c + rNow * (-s);
      float dy = dr_dphi * s + rNow * ( c);
      vec3 dirOut = normalize(a * dx + b * dy);
      float lensStrength = clamp01((12.0 * M - minR) / (10.0 * M)) * bhPhase;
      return vec4(skyColor(dirOut, lensStrength), 1.0);
    }
  }

  return vec4(0,0,0,1);
}

void main() {
  ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
  if (pix.x >= uResolution.x || pix.y >= uResolution.y) return;

  vec2 uv  = (vec2(pix) + 0.5) / vec2(uResolution);
  vec2 ndc = uv * 2.0 - 1.0;
  ndc.x *= float(uResolution.x) / float(uResolution.y);

  float f = 1.0 / tan(radians(uFovY) * 0.5);

  vec3 right   = uCamBasis[0];
  vec3 up      = uCamBasis[1];
  vec3 forward = uCamBasis[2];

  vec3 rayDir = normalize(forward * f + right * ndc.x + up * ndc.y);

  vec4 col = traceSchwarzschild(uCamPos, rayDir);
  imageStore(img, pix, col);
}
)";

static const char* kFullscreenVS = R"(#version 430
out vec2 vUV;
void main() {
  vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2);
  vUV = p;
  gl_Position = vec4(p*2.0-1.0, 0.0, 1.0);
}
)";

static const char* kFullscreenFS = R"(#version 430
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform float uHudCollapse;

float rectMask(vec2 uv, vec2 mn, vec2 mx) {
  vec2 s = step(mn, uv) * step(uv, mx);
  return s.x * s.y;
}

float ringMask(vec2 uv, vec2 c, float r0, float r1) {
  float d = distance(uv, c);
  return step(r0, d) * step(d, r1);
}

void main() {
  vec3 col = texture(uTex, vUV).rgb;

  float c = clamp(uHudCollapse, 0.0, 1.0);
  vec2 hudMin = vec2(0.02, 0.04);
  vec2 hudMax = vec2(0.30, 0.46);

  float panel = rectMask(vUV, hudMin, hudMax);
  if (panel > 0.5) {
    vec3 panelCol = vec3(0.03, 0.035, 0.05);
    col = mix(col, panelCol, 0.82);

    // Progress bar: star life -> supernova -> black hole
    vec2 p0 = hudMin + vec2(0.02, 0.035);
    vec2 p1 = hudMax - vec2(0.02, 0.36);
    float barBg = rectMask(vUV, p0, p1);
    col = mix(col, vec3(0.12), 0.8 * barBg);
    vec2 pf = vec2(mix(p0.x, p1.x, c), p1.y);
    float barFill = rectMask(vUV, p0, pf);
    vec3 lifeCol = mix(vec3(0.8, 0.2, 0.06), vec3(0.9, 0.85, 0.65), smoothstep(0.50, 0.78, c));
    lifeCol = mix(lifeCol, vec3(0.65, 0.82, 1.0), smoothstep(0.78, 1.0, c));
    col = mix(col, lifeCol, 0.95 * barFill);

    // Layer diagram (outer -> inner shells)
    vec2 center = hudMin + vec2(0.14, 0.19);
    float baseR = 0.125;
    vec3 layerCols[6] = vec3[6](
      vec3(0.95, 0.22, 0.08), // H shell
      vec3(0.95, 0.58, 0.16), // He shell
      vec3(0.70, 0.70, 0.74), // C/O
      vec3(0.38, 0.72, 0.56), // Ne
      vec3(0.42, 0.56, 0.95), // Si
      vec3(0.76, 0.76, 0.80)  // Fe core
    );

    float active = 0.0;
    if (c < 0.18) active = 0.0;
    else if (c < 0.36) active = 1.0;
    else if (c < 0.54) active = 2.0;
    else if (c < 0.68) active = 3.0;
    else if (c < 0.78) active = 4.0;
    else active = 5.0;

    for (int i = 0; i < 6; i++) {
      float o = float(i) * 0.018;
      float m = ringMask(vUV, center, baseR - o - 0.0175, baseR - o);
      float isActive = 1.0 - step(0.5, abs(float(i) - active));
      float pulse = 0.75 + 0.25 * sin(40.0 * vUV.x + 30.0 * vUV.y + c * 30.0);
      vec3 lc = layerCols[i] * (1.0 + 0.45 * isActive * pulse);
      col = mix(col, lc, m * 0.96);
    }

    // Layer legend bars (top=outer layers, bottom=core)
    vec2 l0 = hudMin + vec2(0.19, 0.11);
    for (int i = 0; i < 6; i++) {
      float y = l0.y + float(i) * 0.042;
      vec2 a = vec2(l0.x, y);
      vec2 b = vec2(l0.x + 0.08, y + 0.026);
      float lm = rectMask(vUV, a, b);
      float isActive = 1.0 - step(0.5, abs(float(i) - active));
      vec3 lc = layerCols[i] * (0.85 + 0.5 * isActive);
      col = mix(col, lc, lm);
    }
  }

  FragColor = vec4(col, 1.0);
}
)";

static void recreateOutputTex(GLuint& tex, int w, int h) {
  if (tex) glDeleteTextures(1, &tex);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

int main() {
  if (!glfwInit()) die("Failed to init GLFW");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  int W = 1280, H = 720;
  GLFWwindow* win = glfwCreateWindow(W, H, "Schwarzschild Black Hole + Disk", nullptr, nullptr);
  if (!win) die("Failed to create window");

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  glfwSetCursorPosCallback(win, mouseCallback);
  glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) die("Failed to init GLAD");

  // Dummy VAO required for core-profile draws
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // Output texture
  GLuint tex = 0;
  recreateOutputTex(tex, W, H);

  // Programs
  GLuint compProg = linkProgram({ compileShader(GL_COMPUTE_SHADER, kComputeSrc) });
  GLuint blitProg = linkProgram({
    compileShader(GL_VERTEX_SHADER,   kFullscreenVS),
    compileShader(GL_FRAGMENT_SHADER, kFullscreenFS)
  });

  // Uniform locations (compute)
  auto ul = [&](GLuint prog, const char* name) { return glGetUniformLocation(prog, name); };

  GLint uResolution   = ul(compProg, "uResolution");
  GLint uCamPos       = ul(compProg, "uCamPos");
  GLint uCamBasis     = ul(compProg, "uCamBasis");
  GLint uFovY         = ul(compProg, "uFovY");
  GLint uM            = ul(compProg, "uM");
  GLint uPhiStep      = ul(compProg, "uPhiStep");
  GLint uMaxSteps     = ul(compProg, "uMaxSteps");
  GLint uFarR         = ul(compProg, "uFarR");
  GLint uDiskInnerR   = ul(compProg, "uDiskInnerR");
  GLint uDiskOuterR   = ul(compProg, "uDiskOuterR");
  GLint uDiskHalfThick= ul(compProg, "uDiskHalfThick");
  GLint uDiskBoost    = ul(compProg, "uDiskBoost");
  GLint uSkyMode      = ul(compProg, "uSkyMode");
  GLint uTime         = ul(compProg, "uTime");
  GLint uPlanetOrbitR = ul(compProg, "uPlanetOrbitR");
  GLint uPlanetRadius = ul(compProg, "uPlanetRadius");
  GLint uPlanetOmega  = ul(compProg, "uPlanetOmega");
  GLint uPlanetY      = ul(compProg, "uPlanetY");
  GLint uCollapse     = ul(compProg, "uCollapse");
  GLint uStarRadiusStart = ul(compProg, "uStarRadiusStart");
  GLint uStarRadiusEnd   = ul(compProg, "uStarRadiusEnd");

  // Uniform location (blit)
  GLint uTex = ul(blitProg, "uTex");
  GLint uHudCollapse = ul(blitProg, "uHudCollapse");

  // Timing for consistent movement speed
  double lastT = glfwGetTime();

  // Scene params
  const float M = 1.15f;

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();

    double now = glfwGetTime();
    float dt = (float)(now - lastT);
    lastT = now;
    if (dt > 0.05f) dt = 0.05f;

    // Resize (safe: recreate texture)
    int newW, newH;
    glfwGetFramebufferSize(win, &newW, &newH);
    if (newW != W || newH != H) {
      W = newW; H = newH;
      recreateOutputTex(tex, W, H);
      glViewport(0, 0, W, H);
    }

    // Movement
    const float deg2rad = 0.01745329252f;
    float yawR = gYaw * deg2rad;
    float pitR = gPitch * deg2rad;

    float fx = std::cos(yawR) * std::cos(pitR);
    float fy = std::sin(pitR);
    float fz = std::sin(yawR) * std::cos(pitR);
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx/=fl; fy/=fl; fz/=fl;

    // right = normalize(forward x up(0,1,0))
    float rx = fy*0.0f - fz*1.0f;
    float ry = fz*0.0f - fx*0.0f;
    float rz = fx*1.0f - fy*0.0f;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx/=rl; ry/=rl; rz/=rl;

    float step = gMoveSpeed * dt;

    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) { gCamX += fx*step; gCamY += fy*step; gCamZ += fz*step; }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) { gCamX -= fx*step; gCamY -= fy*step; gCamZ -= fz*step; }
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) { gCamX -= rx*step; gCamY -= ry*step; gCamZ -= rz*step; }
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) { gCamX += rx*step; gCamY += ry*step; gCamZ += rz*step; }
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) { gCamY -= step; }
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) { gCamY += step; }

    // Build camera basis/pos for shader
    float basis[9], pos[3];
    computeCameraBasis(basis, pos);

    // ---- Compute pass ----
    glUseProgram(compProg);
    glUniform2i(uResolution, W, H);
    glUniform3f(uCamPos, pos[0], pos[1], pos[2]);
    glUniformMatrix3fv(uCamBasis, 1, GL_FALSE, basis);
    glUniform1f(uFovY, 55.0f);

    float collapse = std::min((float)(now / 18.0), 1.0f);

    glUniform1f(uM, M);
    glUniform1f(uPhiStep, 0.0026f);    // nicer near strong lensing
    glUniform1i(uMaxSteps, 12000);     // allow more loops -> more arcs
    glUniform1f(uFarR, 220.0f);

    glUniform1f(uDiskInnerR, 3.2f * M);
    glUniform1f(uDiskOuterR, 22.0f * M);
    glUniform1f(uDiskHalfThick, 0.025f * M);
    glUniform1f(uDiskBoost, 2.5f);
    glUniform1i(uSkyMode, 1);          // 1=grid, 0=stars
    glUniform1f(uTime, (float)now);
    glUniform1f(uPlanetOrbitR, 30.0f * M);
    glUniform1f(uPlanetRadius, 1.35f * M);
    glUniform1f(uPlanetOmega, 0.20f);
    glUniform1f(uPlanetY, 0.8f * M);
    glUniform1f(uCollapse, collapse);
    glUniform1f(uStarRadiusStart, 7.0f * M);
    glUniform1f(uStarRadiusEnd, 1.6f * M);

    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    GLuint gx = (GLuint)((W + 15) / 16);
    GLuint gy = (GLuint)((H + 15) / 16);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // ---- Blit pass ----
    glUseProgram(blitProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(uTex, 0);
    glUniform1f(uHudCollapse, collapse);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(win);
  }

  glDeleteProgram(compProg);
  glDeleteProgram(blitProg);
  glDeleteTextures(1, &tex);
  glDeleteVertexArrays(1, &vao);

  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}

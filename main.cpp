// main.cpp — Schwarzschild lensing + animated accretion disk + photon ring glow
// + temporal accumulation (HDR) + BLOOM (threshold + blur + composite)
// + improved sky grid (NO derivatives in compute; resolution-based AA) + subtle stars
// + free-fly camera (WASD + mouse)
//
// Requires OpenGL 4.3+ (compute shaders), GLFW, GLAD.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

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
static float gCamX = 0.0f, gCamY = 0.35f, gCamZ = 12.0f;
static float gYaw = -90.0f, gPitch = 0.0f;
static float gLastX = 0.0f, gLastY = 0.0f;
static bool  gFirstMouse = true;
static float gMouseSens = 0.08f;
static float gMoveSpeed = 5.0f;

// Time / accumulation controls
static int   gFrame = 0;
static float gTimeScale = 1.0f;
static bool  gResetAccum = true;

static void mouseCallback(GLFWwindow*, double xpos, double ypos) {
  if (gFirstMouse) {
    gLastX = (float)xpos;
    gLastY = (float)ypos;
    gFirstMouse = false;
  }
  float xoff = (float)xpos - gLastX;
  float yoff = gLastY - (float)ypos;
  gLastX = (float)xpos;
  gLastY = (float)ypos;

  xoff *= gMouseSens;
  yoff *= gMouseSens;

  gYaw   += xoff;
  gPitch += yoff;

  if (gPitch > 89.0f)  gPitch = 89.0f;
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

  float ux = 0.0f, uy = 1.0f, uz = 0.0f;

  float rx = fy*uz - fz*uy;
  float ry = fz*ux - fx*uz;
  float rz = fx*uy - fy*ux;
  float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
  rx/=rl; ry/=rl; rz/=rl;

  float upx = ry*fz - rz*fy;
  float upy = rz*fx - rx*fz;
  float upz = rx*fy - ry*fx;

  pos3[0] = gCamX; pos3[1] = gCamY; pos3[2] = gCamZ;

  basis9[0]=rx;  basis9[1]=ry;  basis9[2]=rz;
  basis9[3]=upx; basis9[4]=upy; basis9[5]=upz;
  basis9[6]=fx;  basis9[7]=fy;  basis9[8]=fz;
}

// ------------------------- Shaders -------------------------

static const char* kComputeSrc = R"(#version 430
layout(local_size_x=16, local_size_y=16) in;

// Output HDR accumulation image
layout(rgba16f, binding=0) uniform image2D imgOut;

// Previous HDR accumulation (read-only)
layout(binding=1) uniform sampler2D uPrevAccum;

uniform ivec2 uResolution;
uniform float uTime;
uniform int   uFrame;
uniform int   uResetAccum;

// Camera
uniform vec3  uCamPos;
uniform mat3  uCamBasis;
uniform float uFovY;

// Black hole
uniform float uM;

// Ray integration
uniform float uPhiStep;
uniform int   uMaxSteps;
uniform float uFarR;

// Accretion disk
uniform float uDiskInnerR;
uniform float uDiskOuterR;
uniform float uDiskHalfThick;
uniform float uDiskBoost;
uniform int   uSkyMode;        // 0=stars, 1=grid

const float PI = 3.141592653589793;

float clamp01(float x) { return max(0.0, min(1.0, x)); }

// Hash/noise
float hash12(vec2 p){
  vec3 p3  = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float noise(vec2 p){
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash12(i);
  float b = hash12(i+vec2(1,0));
  float c = hash12(i+vec2(0,1));
  float d = hash12(i+vec2(1,1));
  vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

vec3 stars(vec3 dir) {
  float h1 = fract(sin(dot(dir*437.0, vec3(12.9898,78.233,37.719))) * 43758.5453);
  float h2 = fract(sin(dot(dir*911.0, vec3(93.9898,67.345,11.135))) * 15238.5453);

  float s1 = smoothstep(0.9986, 1.0, h1);
  float s2 = smoothstep(0.9990, 1.0, h2);

  float tw1 = 0.65 + 0.35*sin(uTime*2.0 + h1*60.0);
  float tw2 = 0.70 + 0.30*sin(uTime*1.6 + h2*90.0);

  vec3 col = vec3(0.0);
  col += s1 * tw1 * vec3(1.8, 1.8, 2.0);
  col += s2 * tw2 * vec3(1.2, 1.4, 2.1);
  return col;
}

// Improved sky: NO fwidth() (compute-safe). Use resolution-based AA width.
vec3 skyColor(vec3 dir) {
  dir = normalize(dir);

  float t = clamp01(dir.y*0.5 + 0.5);
  vec3 base = mix(vec3(0.010,0.012,0.018), vec3(0.055,0.080,0.140), t);
  base += stars(dir);

  if (uSkyMode == 0) return base;

  float lon = atan(dir.z, dir.x);
  float lat = asin(clamp(dir.y, -1.0, 1.0));
  vec2 uv = vec2(lon/(2.0*PI)+0.5, lat/PI+0.5);

  // Grid density
  float mer = 16.0;
  float par = 8.0;

  // Convert to cell space
  vec2 g = vec2(uv.x * mer, uv.y * par);
  vec2 f = fract(g);
  vec2 d = abs(f - 0.5); // distance to line center in cell space (0..0.5)

  // Approx AA width in cell-space: ~1 pixel in uv => (grid cells per screen pixel)
  // uv-per-pixel ~ 1/min(res); multiply by grid density to get cell-space per pixel.
  float invMinRes = 1.0 / float(min(uResolution.x, uResolution.y));
  vec2 cellPerPix = vec2(mer, par) * invMinRes;

  float w = 1.25 * max(cellPerPix.x, cellPerPix.y); // slightly thicker AA band
  float lineW = 0.035; // base line thickness in cell space

  float lx = 1.0 - smoothstep(lineW, lineW + w, d.x);
  float ly = 1.0 - smoothstep(lineW, lineW + w, d.y);

  float pole = abs(dir.y);
  float poleFade = smoothstep(0.98, 0.55, pole);

  float grid = max(lx, ly) * poleFade;

  vec3 gridCol = vec3(0.75, 0.85, 1.05);
  base += grid * gridCol * 0.11;

  // Major lines (lower frequency)
  vec2 gm = vec2(uv.x * 4.0, uv.y * 2.0);
  vec2 fm = fract(gm);
  vec2 dm = abs(fm - 0.5);

  float wM = 1.4 * w;
  float majorW = 0.050;

  float mx = 1.0 - smoothstep(majorW, majorW + wM, dm.x);
  float my = 1.0 - smoothstep(majorW, majorW + wM, dm.y);

  float major = max(mx, my) * poleFade;
  base += major * gridCol * 0.12;

  return base;
}

bool segmentHitsDisk(vec3 p0, vec3 p1, out vec3 hitPos) {
  float y0 = p0.y, y1 = p1.y;
  if ((y0 > 0.0 && y1 > 0.0) || (y0 < 0.0 && y1 < 0.0)) return false;
  float denom = (y0 - y1);
  if (abs(denom) < 1e-8) return false;

  float t = y0 / denom;
  if (t < 0.0 || t > 1.0) return false;

  vec3 p = mix(p0, p1, t);
  if (abs(p.y) > uDiskHalfThick) return false;

  float r = length(p.xz);
  if (r < uDiskInnerR || r > uDiskOuterR) return false;

  hitPos = p;
  return true;
}

vec3 shadeDisk(vec3 p, vec3 segDir, float M) {
  float r = length(p.xz);

  float t = (r - uDiskInnerR) / max(uDiskOuterR - uDiskInnerR, 1e-6);
  float x = clamp01(t);
  float hot = pow(1.0 - x, 1.8);

  vec3 cool = vec3(0.12, 0.02, 0.01);
  vec3 warm = vec3(7.5,  2.6,  0.40);
  vec3 col  = mix(cool, warm, hot);

  vec3 tangent = normalize(vec3(-p.z, 0.0, p.x));

  float beta  = clamp01(sqrt(M / max(r, 1e-4)) * 0.55) * 0.7;
  float gamma = inversesqrt(1.0 - beta*beta);

  float mu  = dot(tangent, normalize(-segDir));
  float dop = 1.0 / (gamma * (1.0 - beta * mu));
  float beam = pow(dop, 3.0);

  float blu = clamp01((dop - 1.0) * 0.9);
  float red = clamp01((1.0 - dop) * 0.9);
  vec3 blueTint = vec3(0.65, 0.85, 1.35);
  vec3 redTint  = vec3(1.35, 0.80, 0.62);
  col *= mix(vec3(1.0), blueTint, blu);
  col *= mix(vec3(1.0), redTint,  red);

  float rs = 2.0 * M;
  float g  = sqrt(max(1.0 - rs / max(r, rs*1.001), 0.0));
  col *= (0.22 + 0.78 * g);

  float ang = atan(p.z, p.x);
  float spiral = sin(ang * 10.0 - r * 0.90 + uTime * 2.1);
  spiral = 0.5 + 0.5 * spiral;

  float n = noise(vec2(ang * 2.2, r * 0.28) + vec2(uTime * 0.55, -uTime * 0.25));
  float grain = mix(0.82, 1.30, n);

  float band = mix(0.70, 1.45, spiral);
  col *= band * grain;

  col *= uDiskBoost * beam;
  return col;
}

vec3 traceSchwarzschild(vec3 camPos, vec3 rayDir) {
  float M  = uM;
  float rs = 2.0 * M;

  vec3 r0 = camPos;
  float rObs = length(r0);
  if (rObs <= rs * 1.001) return vec3(0.0);

  vec3 a = normalize(r0);
  vec3 v = normalize(rayDir);

  vec3 n = cross(r0, v);
  float nlen = length(n);
  if (nlen < 1e-6) return skyColor(v);
  n /= nlen;

  vec3 b = normalize(cross(n, a));

  float v_r = dot(v, a);
  float v_t = dot(v, b);
  if (v_t < 0.0) { b = -b; v_t = -v_t; }

  float L = max(rObs * v_t, 1e-6);

  float u = 1.0 / rObs;
  float p = -(v_r) / L;
  float phi = 0.0;

  vec3 posPrev = (a*cos(phi) + b*sin(phi)) * rObs;

  float rMin = 1e30;

  for (int i = 0; i < uMaxSteps; i++) {
    float r = 1.0 / u;
    if (r <= rs * 1.001) return vec3(0.0);

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

    if (u <= 0.0) return skyColor(v);

    float rNow = 1.0 / u;
    rMin = min(rMin, rNow);

    vec3 posNow = (a*cos(phi) + b*sin(phi)) * rNow;

    vec3 hitPos;
    if (segmentHitsDisk(posPrev, posNow, hitPos)) {
      vec3 segDir = normalize(posNow - posPrev);
      return shadeDisk(hitPos, segDir, M);
    }
    posPrev = posNow;

    float dr_dlambda = -L * p;
    if (rNow >= uFarR && dr_dlambda > 0.0) {
      float dr_dphi = -p / (u*u);
      float c = cos(phi);
      float s = sin(phi);
      float dx = dr_dphi * c + rNow * (-s);
      float dy = dr_dphi * s + rNow * ( c);
      vec3 dirOut = normalize(a * dx + b * dy);

      vec3 base = skyColor(dirOut);

      float photon = 3.0 * M;
      float glow = exp(-abs(rMin - photon) / (0.15 * M));
      base += glow * vec3(0.50, 0.70, 1.10) * 0.35;

      float pole = pow(abs(dirOut.y), 10.0);
      base += pole * (0.06 + 0.05*sin(uTime*3.0)) * vec3(0.5, 0.7, 1.2);

      return base;
    }
  }

  return vec3(0.0);
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

  vec3 hdr = traceSchwarzschild(uCamPos, rayDir);

  vec3 prev = texture(uPrevAccum, uv).rgb;
  bool reset = (uResetAccum == 1) || (uFrame == 0);
  float w = reset ? 1.0 : (1.0 / float(uFrame + 1));
  vec3 accum = mix(prev, hdr, w);

  imageStore(imgOut, pix, vec4(accum, 1.0));
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

static const char* kBloomDownsampleFS = R"(#version 430
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;
uniform vec2 uInvSceneSize;
uniform float uThreshold;

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  vec2 texel = uInvSceneSize;

  vec3 c00 = texture(uScene, vUV + texel*vec2(-0.5,-0.5)).rgb;
  vec3 c10 = texture(uScene, vUV + texel*vec2( 0.5,-0.5)).rgb;
  vec3 c01 = texture(uScene, vUV + texel*vec2(-0.5, 0.5)).rgb;
  vec3 c11 = texture(uScene, vUV + texel*vec2( 0.5, 0.5)).rgb;

  vec3 c = 0.25 * (c00 + c10 + c01 + c11);

  float l = luminance(c);
  float soft = max(l - uThreshold, 0.0);
  vec3 bright = (l > 1e-6) ? (c * (soft / l)) : vec3(0.0);

  FragColor = vec4(bright, 1.0);
}
)";

static const char* kBloomBlurFS = R"(#version 430
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;
uniform vec2 uDirection;
uniform vec2 uInvSize;

void main() {
  vec2 off = uDirection * uInvSize;

  vec3 c = vec3(0.0);
  c += texture(uTex, vUV + off * -4.0).rgb * 0.016216;
  c += texture(uTex, vUV + off * -3.0).rgb * 0.054054;
  c += texture(uTex, vUV + off * -2.0).rgb * 0.121622;
  c += texture(uTex, vUV + off * -1.0).rgb * 0.194594;
  c += texture(uTex, vUV).rgb               * 0.227027;
  c += texture(uTex, vUV + off *  1.0).rgb * 0.194594;
  c += texture(uTex, vUV + off *  2.0).rgb * 0.121622;
  c += texture(uTex, vUV + off *  3.0).rgb * 0.054054;
  c += texture(uTex, vUV + off *  4.0).rgb * 0.016216;

  FragColor = vec4(c, 1.0);
}
)";

static const char* kCompositeFS = R"(#version 430
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uExposure;

vec3 tonemapACES(vec3 x) {
  x *= uExposure;
  vec3 a = x * (2.51*x + 0.03);
  vec3 b = x * (2.43*x + 0.59) + 0.14;
  return clamp(a / b, 0.0, 1.0);
}

void main() {
  vec3 scene = texture(uScene, vUV).rgb;
  vec3 bloom = texture(uBloom, vUV).rgb;

  vec3 hdr = scene + bloom * uBloomStrength;
  vec3 ldr = tonemapACES(hdr);
  ldr = pow(ldr, vec3(1.0/2.2));

  FragColor = vec4(ldr, 1.0);
}
)";

// ------------------------- GL helpers -------------------------

static void recreateHDRTex(GLuint& tex, int w, int h) {
  if (tex) glDeleteTextures(1, &tex);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, w, h);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static void recreateBloomTex(GLuint& tex, int w, int h) {
  if (tex) glDeleteTextures(1, &tex);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, w, h);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static GLuint makeFBOWithColorTex(GLuint colorTex) {
  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
  GLenum draw = GL_COLOR_ATTACHMENT0;
  glDrawBuffers(1, &draw);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    die("FBO not complete");
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return fbo;
}

int main() {
  if (!glfwInit()) die("Failed to init GLFW");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  int W = 1280, H = 720;
  GLFWwindow* win = glfwCreateWindow(W, H, "Black Hole (HDR Accum + Bloom)", nullptr, nullptr);
  if (!win) die("Failed to create window");

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  glfwSetCursorPosCallback(win, mouseCallback);
  glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) die("Failed to init GLAD");

  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // HDR accumulation ping-pong textures
  GLuint accumA = 0, accumB = 0;
  recreateHDRTex(accumA, W, H);
  recreateHDRTex(accumB, W, H);

  // Bloom textures (half-res)
  int bW = std::max(1, W / 2);
  int bH = std::max(1, H / 2);
  GLuint bloomA = 0, bloomB = 0;
  recreateBloomTex(bloomA, bW, bH);
  recreateBloomTex(bloomB, bW, bH);
  GLuint fboBloomA = makeFBOWithColorTex(bloomA);
  GLuint fboBloomB = makeFBOWithColorTex(bloomB);

  // Programs
  GLuint compProg = linkProgram({ compileShader(GL_COMPUTE_SHADER, kComputeSrc) });
  GLuint downProg = linkProgram({ compileShader(GL_VERTEX_SHADER, kFullscreenVS),
                                  compileShader(GL_FRAGMENT_SHADER, kBloomDownsampleFS) });
  GLuint blurProg = linkProgram({ compileShader(GL_VERTEX_SHADER, kFullscreenVS),
                                  compileShader(GL_FRAGMENT_SHADER, kBloomBlurFS) });
  GLuint finalProg = linkProgram({ compileShader(GL_VERTEX_SHADER, kFullscreenVS),
                                   compileShader(GL_FRAGMENT_SHADER, kCompositeFS) });

  auto ul = [&](GLuint prog, const char* name) { return glGetUniformLocation(prog, name); };

  // Compute uniforms
  GLint uResolution   = ul(compProg, "uResolution");
  GLint uTime         = ul(compProg, "uTime");
  GLint uFrame        = ul(compProg, "uFrame");
  GLint uResetAccumU  = ul(compProg, "uResetAccum");
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
  GLint uPrevAccumLoc = ul(compProg, "uPrevAccum");

  // Downsample uniforms
  GLint d_uScene       = ul(downProg, "uScene");
  GLint d_uInvSceneSz  = ul(downProg, "uInvSceneSize");
  GLint d_uThreshold   = ul(downProg, "uThreshold");

  // Blur uniforms
  GLint b_uTex      = ul(blurProg, "uTex");
  GLint b_uDir      = ul(blurProg, "uDirection");
  GLint b_uInvSize  = ul(blurProg, "uInvSize");

  // Final uniforms
  GLint f_uScene          = ul(finalProg, "uScene");
  GLint f_uBloom          = ul(finalProg, "uBloom");
  GLint f_uBloomStrength  = ul(finalProg, "uBloomStrength");
  GLint f_uExposure       = ul(finalProg, "uExposure");

  double lastT = glfwGetTime();
  double simTime = 0.0;

  // Camera-change detection
  static float lastCamX=gCamX, lastCamY=gCamY, lastCamZ=gCamZ;
  static float lastYaw=gYaw, lastPitch=gPitch;
  auto cameraChanged = [&](){
    float dp = std::abs(gCamX-lastCamX)+std::abs(gCamY-lastCamY)+std::abs(gCamZ-lastCamZ);
    float dr = std::abs(gYaw-lastYaw)+std::abs(gPitch-lastPitch);
    bool ch = (dp > 1e-4f) || (dr > 1e-4f);
    lastCamX=gCamX; lastCamY=gCamY; lastCamZ=gCamZ;
    lastYaw=gYaw; lastPitch=gPitch;
    return ch;
  };

  const float M = 1.0f;

  // Bloom knobs
  float bloomThreshold = 1.25f;
  float bloomStrength  = 0.85f;
  float exposure       = 1.05f;

  // Sky mode toggle
  int skyMode = 1; // grid by default

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();

    double now = glfwGetTime();
    float dt = (float)(now - lastT);
    lastT = now;
    if (dt > 0.05f) dt = 0.05f;

    // Time controls
    if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) { gTimeScale = 0.25f; }
    if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) { gTimeScale = 1.0f; }
    if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) { gTimeScale = 3.0f; }
    if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS) { gTimeScale = 0.0f; }

    // Sky toggle: G = grid, H = stars
    static int lastG = GLFW_RELEASE, lastH = GLFW_RELEASE;
    int nowG = glfwGetKey(win, GLFW_KEY_G);
    int nowH = glfwGetKey(win, GLFW_KEY_H);
    if (nowG == GLFW_PRESS && lastG == GLFW_RELEASE) { skyMode = 1; gResetAccum = true; gFrame = 0; }
    if (nowH == GLFW_PRESS && lastH == GLFW_RELEASE) { skyMode = 0; gResetAccum = true; gFrame = 0; }
    lastG = nowG; lastH = nowH;

    simTime += dt * gTimeScale;

    // Resize
    int newW, newH;
    glfwGetFramebufferSize(win, &newW, &newH);
    if (newW != W || newH != H) {
      W = newW; H = newH;
      recreateHDRTex(accumA, W, H);
      recreateHDRTex(accumB, W, H);

      bW = std::max(1, W / 2);
      bH = std::max(1, H / 2);
      recreateBloomTex(bloomA, bW, bH);
      recreateBloomTex(bloomB, bW, bH);

      glDeleteFramebuffers(1, &fboBloomA);
      glDeleteFramebuffers(1, &fboBloomB);
      fboBloomA = makeFBOWithColorTex(bloomA);
      fboBloomB = makeFBOWithColorTex(bloomB);

      glViewport(0, 0, W, H);
      gResetAccum = true;
      gFrame = 0;
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

    float rx = -fz;
    float ry = 0.0f;
    float rz =  fx;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx/=rl; ry/=rl; rz/=rl;

    float step = gMoveSpeed * dt;

    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) { gCamX += fx*step; gCamY += fy*step; gCamZ += fz*step; }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) { gCamX -= fx*step; gCamY -= fy*step; gCamZ -= fz*step; }
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) { gCamX -= rx*step; gCamY -= ry*step; gCamZ -= rz*step; }
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) { gCamX += rx*step; gCamY += ry*step; gCamZ += rz*step; }
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) { gCamY -= step; }
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) { gCamY += step; }

    if (cameraChanged()) {
      gResetAccum = true;
      gFrame = 0;
    }

    float basis[9], pos[3];
    computeCameraBasis(basis, pos);

    GLuint writeAccum = (gFrame % 2 == 0) ? accumA : accumB;
    GLuint prevAccum  = (gFrame % 2 == 0) ? accumB : accumA;

    // -------------------- Compute pass --------------------
    glUseProgram(compProg);

    glUniform2i(uResolution, W, H);
    glUniform1f(uTime, (float)simTime);
    glUniform1i(uFrame, gFrame);
    glUniform1i(uResetAccumU, gResetAccum ? 1 : 0);

    glUniform3f(uCamPos, pos[0], pos[1], pos[2]);
    glUniformMatrix3fv(uCamBasis, 1, GL_FALSE, basis);
    glUniform1f(uFovY, 55.0f);

    glUniform1f(uM, M);
    glUniform1f(uPhiStep, 0.0026f);
    glUniform1i(uMaxSteps, 12000);
    glUniform1f(uFarR, 220.0f);

    glUniform1f(uDiskInnerR, 3.2f * M);
    glUniform1f(uDiskOuterR, 22.0f * M);
    glUniform1f(uDiskHalfThick, 0.025f * M);
    glUniform1f(uDiskBoost, 2.6f);
    glUniform1i(uSkyMode, skyMode);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, prevAccum);
    glUniform1i(uPrevAccumLoc, 1);

    glBindImageTexture(0, writeAccum, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    GLuint gx = (GLuint)((W + 15) / 16);
    GLuint gy = (GLuint)((H + 15) / 16);
    glDispatchCompute(gx, gy, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // -------------------- Bloom downsample+threshold --------------------
    glBindFramebuffer(GL_FRAMEBUFFER, fboBloomA);
    glViewport(0, 0, bW, bH);

    glUseProgram(downProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, writeAccum);
    glUniform1i(d_uScene, 0);
    glUniform2f(d_uInvSceneSz, 1.0f / (float)W, 1.0f / (float)H);
    glUniform1f(d_uThreshold, bloomThreshold);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // -------------------- Bloom blur H --------------------
    glBindFramebuffer(GL_FRAMEBUFFER, fboBloomB);
    glUseProgram(blurProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, bloomA);
    glUniform1i(b_uTex, 0);
    glUniform2f(b_uDir, 1.0f, 0.0f);
    glUniform2f(b_uInvSize, 1.0f / (float)bW, 1.0f / (float)bH);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // -------------------- Bloom blur V --------------------
    glBindFramebuffer(GL_FRAMEBUFFER, fboBloomA);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, bloomB);
    glUniform1i(b_uTex, 0);
    glUniform2f(b_uDir, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // -------------------- Final composite --------------------
    glViewport(0, 0, W, H);
    glUseProgram(finalProg);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, writeAccum);
    glUniform1i(f_uScene, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, bloomA);
    glUniform1i(f_uBloom, 1);

    glUniform1f(f_uBloomStrength, bloomStrength);
    glUniform1f(f_uExposure, exposure);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(win);

    gResetAccum = false;
    gFrame++;
  }

  glDeleteProgram(compProg);
  glDeleteProgram(downProg);
  glDeleteProgram(blurProg);
  glDeleteProgram(finalProg);

  glDeleteFramebuffers(1, &fboBloomA);
  glDeleteFramebuffers(1, &fboBloomB);

  glDeleteTextures(1, &accumA);
  glDeleteTextures(1, &accumB);
  glDeleteTextures(1, &bloomA);
  glDeleteTextures(1, &bloomB);

  glDeleteVertexArrays(1, &vao);

  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}

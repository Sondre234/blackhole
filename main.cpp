// main.cpp — Kerr lensing (ZAMO tetrad ray->(xi,eta)) + improved disk shading
// + temporal accumulation (HDR) + BLOOM (threshold + blur + composite)
// + improved sky grid (compute-safe AA) + subtle stars
// + free-fly camera (WASD + mouse)
//
// Fixes included:
// - Kerr turning-point handling (flip sr/sth when R or Theta reaches ~0)
// - Proper escape test using dr>0 instead of a position-delta hack
// - Clamp disk redshift/beaming to avoid “tan slab” blowouts
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

// ------------------------- Camera -------------------------
static float gCamX = 0.0f, gCamY = 1.8f, gCamZ = 12.0f;
static float gYaw = -90.0f, gPitch = -8.0f;
static bool  gLockCamera = true;
static float gLastX = 0.0f, gLastY = 0.0f;
static bool  gFirstMouse = true;
static float gMouseSens = 0.08f;
static float gMoveSpeed = 5.0f;

// Time / accumulation controls
static int   gFrame = 0;
static float gTimeScale = 1.0f;
static bool  gResetAccum = true;

static void mouseCallback(GLFWwindow*, double xpos, double ypos) {
  if (gLockCamera) return;

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

// ------------------------- Kerr ISCO (C++) -------------------------
static float kerrISCO(float M, float a) {
  float am = a / M;
  float one = 1.0f;
  float z1 = one + std::cbrt(one - am*am) * (std::cbrt(one + am) + std::cbrt(one - am));
  float z2 = std::sqrt(3.0f*am*am + z1*z1);
  float s  = (a >= 0.0f) ? 1.0f : -1.0f;
  float t  = (3.0f - z1) * (3.0f + z1 + 2.0f*z2);
  t = std::max(t, 0.0f);
  return M * (3.0f + z2 - s * std::sqrt(t));
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
uniform vec3  uCamPos;      // world/cartesian
uniform mat3  uCamBasis;    // world
uniform float uFovY;

// Black hole
uniform float uM;
uniform float uA;           // Kerr spin parameter a (|a|<=M)

// Ray integration
uniform float uLambdaStep;  // affine step
uniform int   uMaxSteps;
uniform float uFarR;

// Accretion disk
uniform float uDiskInnerR;
uniform float uDiskOuterR;
uniform float uDiskHalfThick;
uniform float uDiskBoost;
uniform int   uSkyMode;        // 0=stars, 1=grid
uniform vec3  uLightDir;       // moving directional key light

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

  float mer = 16.0;
  float par = 8.0;

  vec2 g = vec2(uv.x * mer, uv.y * par);
  vec2 f = fract(g);
  vec2 d = abs(f - 0.5);

  float invMinRes = 1.0 / float(min(uResolution.x, uResolution.y));
  vec2 cellPerPix = vec2(mer, par) * invMinRes;

  float w = 1.25 * max(cellPerPix.x, cellPerPix.y);
  float lineW = 0.035;

  float lx = 1.0 - smoothstep(lineW, lineW + w, d.x);
  float ly = 1.0 - smoothstep(lineW, lineW + w, d.y);

  float pole = abs(dir.y);
  float poleFade = smoothstep(0.98, 0.55, pole);

  float grid = max(lx, ly) * poleFade;

  vec3 gridCol = vec3(0.75, 0.85, 1.05);
  base += grid * gridCol * 0.11;

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

// --- Kerr metric helpers (Boyer–Lindquist) ---
void kerr_metric(float r, float th, float M, float a,
                 out float Sigma, out float Delta, out float A,
                 out float g_tt, out float g_tphi, out float g_rr, out float g_thth, out float g_phiphi)
{
  float ct = cos(th);
  float st = max(sin(th), 1e-6);

  Sigma = r*r + a*a*ct*ct;
  Delta = r*r - 2.0*M*r + a*a;

  float rp2 = r*r + a*a;
  A = rp2*rp2 - a*a*Delta*st*st;

  g_tt     = -(1.0 - (2.0*M*r)/Sigma);
  g_tphi   = -(2.0*M*a*r*st*st)/Sigma;
  g_rr     = Sigma / max(Delta, 1e-8);
  g_thth   = Sigma;
  g_phiphi = (A*st*st)/Sigma;
}

void cart_to_bl(vec3 x, out float r, out float th, out float ph)
{
  float a = uA;

  float xx = x.x, yy = x.y, zz = x.z;
  float rho2 = xx*xx + yy*yy + zz*zz;

  float k = rho2 - a*a;
  float disc = k*k + 4.0*a*a*yy*yy;

  float r2 = 0.5 * (k + sqrt(max(disc, 0.0)));
  r = sqrt(max(r2, 1e-12));

  ph = atan(zz, xx);

  float ct = clamp(yy / max(r, 1e-8), -1.0, 1.0);
  th = acos(ct);
}


void local_spherical_basis(float th, float ph, out vec3 e_r, out vec3 e_th, out vec3 e_ph)
{
  float st = sin(th), ct = cos(th);
  float sp = sin(ph), cp = cos(ph);

  e_r  = vec3(st*cp,  ct, st*sp);
  e_th = vec3(ct*cp, -st, ct*sp);
  e_ph = vec3(-sp,   0.0,  cp);
}

vec3 bl_to_xyz(float r, float th, float ph) {
  float a = uA;
  float st = sin(th), ct = cos(th);

  float R = sqrt(max(r*r + a*a, 0.0)); // sqrt(r^2 + a^2)

  float x = R * st * cos(ph);
  float y = r * ct;
  float z = R * st * sin(ph);
  return vec3(x, y, z);
}

struct InitConsts {
  float r, th, ph;
  float xi;
  float eta;
  float sr;
  float sth;
};

// ZAMO tetrad mapping: world ray -> Kerr constants (xi=Lz/E, eta=Q/E^2)
InitConsts kerr_init_from_ray_ZAMO(vec3 camPos, vec3 rayDir)
{
  InitConsts C;
  float r, th, ph;
  cart_to_bl(camPos, r, th, ph);

  float M = uM, a = uA;

  float Sigma, Delta, A, g_tt, g_tphi, g_rr, g_thth, g_phiphi;
  kerr_metric(r, th, M, a, Sigma, Delta, A, g_tt, g_tphi, g_rr, g_thth, g_phiphi);

  vec3 e_r, e_th, e_ph;
  local_spherical_basis(th, ph, e_r, e_th, e_ph);

  float nr  = dot(rayDir, e_r);
  float nth = dot(rayDir, e_th);
  float nph = dot(rayDir, e_ph);

  float omega = -g_tphi / max(g_phiphi, 1e-8);
  float alpha = sqrt(max(Sigma * Delta / max(A, 1e-8), 1e-8));

  float nt = 1.0 / alpha;          // n^t
  float nphi_t = omega / alpha;    // n^phi

  float er_r   = sqrt(max(Delta / Sigma, 0.0));
  float eth_th = 1.0 / sqrt(max(Sigma, 1e-8));
  float eph_ph = 1.0 / sqrt(max(g_phiphi, 1e-8));

  // contravariant p^mu
  float pt   = nt;
  float pr   = nr * er_r;
  float pth  = nth * eth_th;
  float pphi = nphi_t + nph * eph_ph;

  // covariant p_mu
  float p_t   = g_tt*pt + g_tphi*pphi;
  float p_phi = g_tphi*pt + g_phiphi*pphi;
  float p_th  = g_thth*pth;

  float E  = -p_t;
  float Lz =  p_phi;

  float st = max(sin(th), 1e-6);
  float ct = cos(th);

  // Null-geodesic Carter constant:
  // Q = p_theta^2 + cos^2(theta) * (Lz^2/sin^2(theta) - a^2 E^2)
  float Q = p_th*p_th + (ct*ct) * ((Lz*Lz)/(st*st) - a*a*E*E);

  float sr  = (pr  >= 0.0) ? 1.0 : -1.0;
  float sth = (pth >= 0.0) ? 1.0 : -1.0;

  C.r = r; C.th = th; C.ph = ph;
  C.xi = Lz / max(E, 1e-8);
  C.eta = Q  / max(E*E, 1e-8);
  C.sr = sr;
  C.sth = sth;
  return C;
}

struct KerrState {
  float r, th, ph;
  float xi;
  float eta;
  float sr;
  float sth;
};

// derivative + raw potentials (for turning-point detection)
void kerr_deriv(in KerrState s,
                out float dr, out float dth, out float dph,
                out float Rraw, out float Thraw)
{
  float M = uM;
  float a = uA;

  float r = s.r;
  float th = s.th;

  float ct = cos(th);
  float st = max(sin(th), 1e-6);

  float Sigma = r*r + a*a*ct*ct;
  float Delta = r*r - 2.0*M*r + a*a;

  float xi  = s.xi;
  float eta = s.eta;

  float P = (r*r + a*a) - a*xi;

  Rraw  = P*P - Delta*(eta + (xi - a)*(xi - a));
  // Polar potential for null Kerr geodesics:
  // Theta = eta + a^2 cos^2(theta) - xi^2 cot^2(theta)
  Thraw = eta + a*a*ct*ct - (xi*xi) * (ct*ct)/(st*st);

  float R  = max(Rraw,  0.0);
  float Th = max(Thraw, 0.0);

  dr  = s.sr  * sqrt(R)  / max(Sigma, 1e-8);
  dth = s.sth * sqrt(Th) / max(Sigma, 1e-8);

  float term1 = a * P / (max(Delta, 1e-8) * max(Sigma, 1e-8));
  float term2 = (xi/(st*st) - a) / max(Sigma, 1e-8);
  dph = term1 + term2;
}

void kerr_step(inout KerrState s, float h) {
  float dr1, dt1, dp1, R1, T1; kerr_deriv(s, dr1, dt1, dp1, R1, T1);

  KerrState s2 = s; s2.r += 0.5*h*dr1; s2.th += 0.5*h*dt1; s2.ph += 0.5*h*dp1;
  float dr2, dt2, dp2, R2, T2; kerr_deriv(s2, dr2, dt2, dp2, R2, T2);

  KerrState s3 = s; s3.r += 0.5*h*dr2; s3.th += 0.5*h*dt2; s3.ph += 0.5*h*dp2;
  float dr3, dt3, dp3, R3, T3; kerr_deriv(s3, dr3, dt3, dp3, R3, T3);

  KerrState s4 = s; s4.r += h*dr3; s4.th += h*dt3; s4.ph += h*dp3;
  float dr4, dt4, dp4, R4, T4; kerr_deriv(s4, dr4, dt4, dp4, R4, T4);

  s.r  += (h/6.0)*(dr1 + 2.0*dr2 + 2.0*dr3 + dr4);
  s.th += (h/6.0)*(dt1 + 2.0*dt2 + 2.0*dt3 + dt4);
  s.ph += (h/6.0)*(dp1 + 2.0*dp2 + 2.0*dp3 + dp4);

  s.th = clamp(s.th, 1e-4, PI-1e-4);

  // --- TURNING POINT FIX ---
  // flip sr/sth near turning points so rays bounce correctly
  float epsR  = 1e-7 * (1.0 + 10.0*h);
  float epsTh = 1e-7 * (1.0 + 10.0*h);

  float dr, dth, dph, Rraw, Thraw;
  kerr_deriv(s, dr, dth, dph, Rraw, Thraw);

  if (Rraw  <= epsR)  s.sr  *= -1.0;
  if (Thraw <= epsTh) s.sth *= -1.0;
}

// --- Improved disk shading ---
float diskFlux(float r, float rin) {
  float x = max(r, rin*1.0005);
  float f = pow(x, -3.0) * (1.0 - sqrt(rin / x));
  return max(f, 0.0);
}

vec3 bbApprox(float t) {
  t = clamp01(t);
  vec3 cool = vec3(0.28, 0.06, 0.02);
  vec3 mid  = vec3(1.00, 0.36, 0.08);
  vec3 hot  = vec3(2.30, 1.05, 0.32);
  vec3 a = mix(cool, mid, smoothstep(0.0, 0.62, t));
  return mix(a, hot, smoothstep(0.58, 1.0, t));
}

float gravFactorKerr(float r, float M, float a) {
  float Delta = r*r - 2.0*M*r + a*a;
  return sqrt(max(Delta / max(r*r + a*a, 1e-6), 0.0));
}

vec3 shadeDiskBetter(vec3 p, vec3 segDir) {
  float M = uM, a = uA;
  float r = length(p.xz);

  float rin = uDiskInnerR;
  float F = diskFlux(r, rin);
  float T = pow(F, 0.25);
  T = clamp(T * 1.55, 0.0, 1.0);

  vec3 col = bbApprox(T);

  float rinN = r / max(rin, 1e-4);
  float innerRim = 1.0 + 1.4 * exp(-pow((rinN - 1.25) / 0.33, 2.0));
  float outerFalloff = exp(-0.06 * max(r - rin, 0.0));
  col *= innerRim * outerFalloff;

  // Spin-influenced Keplerian proxy: Omega ~ 1 / (r^(3/2) + a)
  float Omega = 1.0 / (pow(max(r, 1e-3), 1.5) + a);
  float v = clamp(Omega * r, 0.0, 0.75);

  float beta  = v;
  float gamma = inversesqrt(1.0 - beta*beta);

  vec3 tangent = normalize(vec3(-p.z, 0.0, p.x));
  float mu = dot(tangent, normalize(-segDir));

  float D = 1.0 / (gamma * (1.0 - beta * mu));
  float ggrav = gravFactorKerr(r, M, a);
  float g = D * ggrav;

  // Clamp beaming so it doesn't explode into "tan slab"
  g = clamp(g, 0.0, 3.0);
  float beam = min(g*g*g, 30.0);

  // limb darkening (mild)
  vec3 n = vec3(0.0, 1.0, 0.0);
  float cosi = clamp(dot(n, normalize(-segDir)), 0.0, 1.0);
  float limb = 0.55 + 0.45*cosi;

  // moving key-light sweep across the disk so accretion structure is easier to read
  vec2 pDir = normalize(p.xz + vec2(1e-6));
  vec2 lDir = normalize(uLightDir.xz + vec2(1e-6));
  float lightSweep = 0.35 + 0.65*pow(clamp(0.5 + 0.5*dot(pDir, lDir), 0.0, 1.0), 1.3);

  col *= beam * limb * lightSweep;

  // subtle texture structure
  float ang = atan(p.z, p.x);
  float spiral = 0.5 + 0.5*sin(ang*8.0 - r*0.85 + uTime*1.8);
  float nse = noise(vec2(ang*2.0, r*0.25) + vec2(uTime*0.4, -uTime*0.2));
  float textureMod = mix(0.85, 1.25, spiral) * mix(0.90, 1.15, nse);
  col *= textureMod;

  col *= uDiskBoost;
  return col;
}

// Kerr photon ring (equatorial) radii for glow weighting (cheap, pretty)
float photonRadiusEq(float M, float a, float sgn) {
  float x = clamp(-sgn * a / max(M, 1e-8), -1.0, 1.0);
  float ang = acos(x);
  return 2.0*M*(1.0 + cos((2.0/3.0)*ang));
}

// Trace Kerr: returns HDR color
vec3 traceKerr(vec3 camPos, vec3 rayDir) {
  float M = uM;
  float a = uA;

  float rplus = M + sqrt(max(M*M - a*a, 0.0));

  InitConsts C = kerr_init_from_ray_ZAMO(camPos, rayDir);

  KerrState s;
  s.r = C.r;
  s.th = C.th;
  s.ph = C.ph;
  s.xi = C.xi;
  s.eta = C.eta;
  s.sr = C.sr;
  s.sth = C.sth;

  if (s.r <= rplus*1.001) return vec3(0.0);

  vec3 posPrev = bl_to_xyz(s.r, s.th, s.ph);

  float rMin = 1e30;

  for (int i = 0; i < uMaxSteps; i++) {
    if (s.r <= rplus*1.001) return vec3(0.0);

    kerr_step(s, uLambdaStep);

    float rNow = s.r;
    rMin = min(rMin, rNow);

    vec3 posNow = bl_to_xyz(s.r, s.th, s.ph);

    vec3 hitPos;
    if (segmentHitsDisk(posPrev, posNow, hitPos)) {
      vec3 segDir = normalize(posNow - posPrev);
      return shadeDiskBetter(hitPos, segDir);
    }

    // Proper escape test: use dr>0 at large r
    float dr, dth, dph, Rraw, Thraw;
    kerr_deriv(s, dr, dth, dph, Rraw, Thraw);

    if (rNow >= uFarR && dr > 0.0) {
      vec3 dirOut = normalize(posNow - posPrev);
      vec3 base = skyColor(dirOut);

      float rph_p = photonRadiusEq(M, a, +1.0);
      float rph_m = photonRadiusEq(M, a, -1.0);
      float d = min(abs(rMin - rph_p), abs(rMin - rph_m));
      float glow = exp(-d / (0.18 * M));
      base += glow * vec3(0.30, 0.48, 0.95) * 0.24;

      // sharpen shadow boundary so the black-hole silhouette reads clearly
      float shadowEdge = smoothstep(2.2*M, 3.9*M, rMin);
      base *= mix(0.12, 1.0, shadowEdge);
      float rim = exp(-pow((rMin - 2.95*M) / (0.33*M), 2.0));
      base += rim * vec3(0.95, 0.55, 0.20) * 0.22;

      float pole = pow(abs(dirOut.y), 10.0);
      base += pole * (0.06 + 0.05*sin(uTime*3.0)) * vec3(0.5, 0.7, 1.2);

      return base;
    }

    posPrev = posNow;
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

  vec3 hdr = traceKerr(uCamPos, rayDir);

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
  GLFWwindow* win = glfwCreateWindow(W, H, "Kerr Black Hole (HDR Accum + Bloom)", nullptr, nullptr);
  if (!win) die("Failed to create window");

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  glfwSetCursorPosCallback(win, mouseCallback);
  glfwSetInputMode(win, GLFW_CURSOR, gLockCamera ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

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
  GLint uA            = ul(compProg, "uA");
  GLint uLambdaStep   = ul(compProg, "uLambdaStep");
  GLint uMaxSteps     = ul(compProg, "uMaxSteps");
  GLint uFarR         = ul(compProg, "uFarR");
  GLint uDiskInnerR   = ul(compProg, "uDiskInnerR");
  GLint uDiskOuterR   = ul(compProg, "uDiskOuterR");
  GLint uDiskHalfThick= ul(compProg, "uDiskHalfThick");
  GLint uDiskBoost    = ul(compProg, "uDiskBoost");
  GLint uSkyMode      = ul(compProg, "uSkyMode");
  GLint uLightDir     = ul(compProg, "uLightDir");
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

  // Spin knobs (keyboard tweakable)
  float a = 0.75f * M;

  // Bloom knobs
  float bloomThreshold = 1.55f;
  float bloomStrength  = 0.48f;
  float exposure       = 0.95f;

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

    // Spin controls: Z/X decrease/increase
    static int lastZ = GLFW_RELEASE, lastX = GLFW_RELEASE;
    int nowZ = glfwGetKey(win, GLFW_KEY_Z);
    int nowX = glfwGetKey(win, GLFW_KEY_X);
    if (nowZ == GLFW_PRESS && lastZ == GLFW_RELEASE) { a -= 0.05f * M; gResetAccum = true; gFrame = 0; }
    if (nowX == GLFW_PRESS && lastX == GLFW_RELEASE) { a += 0.05f * M; gResetAccum = true; gFrame = 0; }
    lastZ = nowZ; lastX = nowX;
    a = std::max(-0.999f*M, std::min(0.999f*M, a));

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

    if (!gLockCamera) {
      if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) { gCamX += fx*step; gCamY += fy*step; gCamZ += fz*step; }
      if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) { gCamX -= fx*step; gCamY -= fy*step; gCamZ -= fz*step; }
      if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) { gCamX -= rx*step; gCamY -= ry*step; gCamZ -= rz*step; }
      if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) { gCamX += rx*step; gCamY += ry*step; gCamZ += rz*step; }
      if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) { gCamY -= step; }
      if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) { gCamY += step; }
    }

    if (cameraChanged()) {
      gResetAccum = true;
      gFrame = 0;
    }

    float basis[9], pos[3];
    computeCameraBasis(basis, pos);

    GLuint writeAccum = (gFrame % 2 == 0) ? accumA : accumB;
    GLuint prevAccum  = (gFrame % 2 == 0) ? accumB : accumA;

    float rin  = kerrISCO(M, a);
    float rout = 22.0f * M;

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
    glUniform1f(uA, a);

    glUniform1f(uLambdaStep, 0.02f);
    glUniform1i(uMaxSteps, 9000);
    glUniform1f(uFarR, 260.0f);

    glUniform1f(uDiskInnerR, rin);
    glUniform1f(uDiskOuterR, rout);
    glUniform1f(uDiskHalfThick, 0.012f * M);
    glUniform1f(uDiskBoost, 1.15f);
    glUniform1i(uSkyMode, skyMode);

    float lightYaw = (float)(simTime * 0.65);
    float lx = std::cos(lightYaw);
    float lz = std::sin(lightYaw);
    glUniform3f(uLightDir, lx, 0.25f, lz);

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

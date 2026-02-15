#version 330 core
out vec4 FragColor;
in vec2 vUV;

uniform vec2  uRes;
uniform float uTime;

// Camera (true 3D raygen)
uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uTanFov;   // tan(fov*0.5)

// Scene params
uniform float uRs;        // black hole radius (sphere at origin)
uniform float uLens;      // placeholder for future bending (unused here)
uniform float uDiskR;     // accretion ring radius
uniform float uDiskW;     // ring half-width (thickness)
uniform float uTilt;      // disk tilt around X (radians)
uniform float uOmega;     // disk angular velocity (rad/s)

uniform int   uShowRing;

// Utility
float hash12(vec2 p){ vec3 p3=fract(vec3(p.xyx)*0.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }

vec3 backdrop(vec3 rd){
    // simple warm starfield depending on ray dir
    float v = pow(max(rd.y*0.5+0.5, 0.0), 1.2);
    vec3 base = mix(vec3(0.02,0.01,0.01), vec3(0.1,0.03,0.01), v);
    // sprinkle
    vec2 uv = rd.xz*0.5 + 0.5;
    float s=0.0; for(int i=0;i<2;i++){ vec2 q=fract(uv*1.7+float(i)*0.173)-0.5; float d=length(q); s+=smoothstep(0.02,0.0,d)*(0.3+0.7*hash12(q*23.0+uTime*0.05)); uv*=1.9; }
    return base + vec3(1.0,0.5,0.1)*s*0.25;
}

// Ray-sphere (origin, radius)
bool hitHorizon(vec3 ro, vec3 rd, float r, out float t){
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r*r;
    float h = b*b - c;
    if(h < 0.0) return false;
    h = sqrt(h);
    float t0 = -b - h;
    float t1 = -b + h;
    t = (t0 > 0.0) ? t0 : ((t1 > 0.0) ? t1 : 1e9);
    return t < 1e8;
}

// Disk plane tilted around X axis
mat3 rotX(float a){ float c=cos(a), s=sin(a); return mat3(1,0,0, 0,c,-s, 0,s,c); }

bool hitDisk(vec3 ro, vec3 rd, float R, float halfW, float tilt, out float t, out vec3 hp, out float ang){
    // plane: normal = rotX(tilt)*vec3(0,1,0), through origin
    vec3 n = rotX(tilt)*vec3(0.0,1.0,0.0);
    float denom = dot(n, rd);
    if (abs(denom) < 1e-5) return false;
    float tt = -dot(n, ro) / denom; // plane at origin
    if (tt <= 0.0) return false;
    vec3 p = ro + rd*tt;
    // project p to disk local frame (tilt inverse)
    mat3 inv = rotX(-tilt);
    vec3 pl = inv * p;
    float r = length(pl.xz);
    if (abs(r - R) > halfW) return false;
    t = tt; hp = p; ang = atan(pl.z, pl.x); // angle in disk local
    return true;
}

void main(){
    // build ray
    float asp = uRes.x/uRes.y;
    vec2 ndc = (vUV*2.0-1.0); ndc.x *= asp;
    vec3 rd = normalize(uCamFwd + ndc.x*uTanFov*uCamRight + ndc.y*uTanFov*uCamUp);
    vec3 ro = uCamPos;

    // background
    vec3 col = backdrop(rd);

    // event horizon
    float tBH; if (hitHorizon(ro, rd, uRs, tBH)) {
        // anything behind horizon is black
        col = mix(col, vec3(0.0), 1.0);
        FragColor = vec4(col,1.0);
        return;
    }

    // disk
    if (uShowRing==1) {
        float t; vec3 hp; float ang;
        if (hitDisk(ro, rd, uDiskR, uDiskW, uTilt, t, hp, ang)){
            // simple emissive with Doppler-ish brightness from tangential velocity
            float phi = ang - uOmega*uTime;
            // tangential dir in local disk frame -> world approx by rotation
            vec3 vdir_local = normalize(vec3(-sin(phi), 0.0, cos(phi)));
            vec3 vdir_world = rotX(uTilt) * vdir_local;
            float toward = clamp(dot(vdir_world, -rd)*0.8 + 0.2, 0.0, 1.0);
            float heat = 0.9 + 0.1*sin(phi*6.0 + uTime*3.0);
            vec3 warmA=vec3(1.0,0.86,0.40), warmB=vec3(0.95,0.35,0.08);
            vec3 ring = mix(warmB, warmA, toward) * heat;
            // composite over bg
            col = mix(col, ring, 0.85);
        }
    }

    // slight vignette
    float vig = smoothstep(1.2, 0.2, length(ndc));
    col *= mix(0.85, 1.0, vig);
    col = pow(col, vec3(0.95));

    FragColor = vec4(col,1.0);
}

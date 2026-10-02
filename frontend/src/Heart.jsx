import { Component, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, useGLTF } from "@react-three/drei";
import * as THREE from "three";

// Model: /heart_coronary.glb (closed anatomical heart, centred at the origin, meshopt + webp compressed).
// Artery centre-lines are in the model's own coordinates, snapped onto the heart surface.
// Edit a point to re-route an artery; keep the ids LAD / LCX / RCA.
const PATHS = {
  LAD: [[9.8,28.0,23.1],[10.3,25.2,25.3],[11.2,22.1,27.5],[12.4,18.6,29.5],[13.6,14.9,31.6],[14.2,11.3,33.9],[14.4,8.7,35.1],[14.7,4.8,36.4],[14.8,-1.5,39.0],[14.8,-6.0,41.8],[14.9,-8.5,44.2],[15.1,-11.4,46.5],[15.6,-14.3,48.2],[16.6,-17.3,49.1],[17.6,-21.1,49.9],[17.9,-25.1,50.7],[17.8,-28.9,51.3],[17.8,-32.9,51.5],[17.6,-37.3,51.7],[17.0,-41.6,52.2],[15.8,-45.7,53.0],[14.3,-49.0,53.6],[12.5,-51.7,53.8],[10.6,-55.2,53.8],[8.2,-58.7,53.8],[6.1,-61.8,53.6],[4.3,-65.1,52.8],[2.6,-68.6,51.6]],
  RCA: [[-5.7,28.1,19.3],[-13.9,26.9,14.9],[-19.7,22.9,14.3],[-24.3,17.3,16.1],[-28.8,11.7,18.5],[-32.0,6.3,21.3],[-33.3,0.9,24.2],[-33.6,-4.8,27.4],[-33.6,-10.9,30.3],[-32.4,-17.8,33.2],[-29.4,-24.5,37.3],[-27.6,-29.7,41.4],[-27.8,-33.8,43.5],[-28.4,-36.8,43.7],[-29.9,-39.4,42.2],[-32.3,-41.8,39.6],[-35.4,-44.7,35.8],[-39.5,-47.7,29.2],[-43.2,-49.2,17.5],[-42.4,-49.4,1.5],[-37.2,-48.1,-12.1],[-31.9,-45.5,-19.5],[-27.3,-42.9,-24.6],[-23.1,-40.1,-31.1],[-19.1,-38.0,-38.4],[-14.7,-37.9,-44.1],[-10.1,-38.1,-48.5],[-4.9,-37.9,-52.7]],
  LCX: [[28.3,25.5,22.1],[33.4,23.4,21.3],[39.1,22.0,20.3],[43.5,20.9,18.1],[46.1,19.3,14.5],[47.4,16.9,10.4],[47.6,15.0,5.5],[48.0,14.1,-0.3],[48.7,13.2,-5.7],[49.0,12.0,-10.5],[48.6,11.1,-15.2],[47.6,10.0,-20.1],[46.1,8.4,-25.6],[44.3,7.3,-30.8],[42.6,6.6,-34.7],[40.5,5.2,-38.4],[38.0,2.9,-42.7],[36.0,0.1,-45.9],[34.7,-2.6,-48.2],[33.2,-5.2,-51.2],[31.8,-8.2,-54.4],[30.6,-11.2,-56.4],[29.0,-14.5,-56.8],[26.0,-18.3,-55.9],[22.1,-22.7,-55.0],[18.1,-25.9,-55.9],[13.6,-28.0,-57.4],[9.0,-30.4,-58.2]],
};
const CENTER = [0, 0, 0];             // model is pre-centred at the origin
const SCALE = 3.2 / 156;                 // model height 156 -> 3.2 scene units
const RADIUS = 1.7;                      // artery tube radius (model units)

const TAPER = 0.6;   // tube radius shrinks to 40% at the distal end

function Vessel({ id, color, selected, onSelect }) {
  const [hover, setHover] = useState(false);
  const mat = useRef();
  const { geo, halo } = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(PATHS[id].map((p) => new THREE.Vector3(...p)));
    const N = 90, R = 12;
    const make = (r) => {                       // tube whose radius tapers along the path
      const g = new THREE.TubeGeometry(curve, N, r, R, false);
      const pos = g.attributes.position, c = new THREE.Vector3(), v = new THREE.Vector3();
      for (let i = 0; i <= N; i++) {
        curve.getPointAt(i / N, c);
        const f = 1 - TAPER * (i / N);
        for (let j = 0; j <= R; j++) {
          const n = i * (R + 1) + j;
          v.fromBufferAttribute(pos, n).sub(c).multiplyScalar(f).add(c);
          pos.setXYZ(n, v.x, v.y, v.z);
        }
      }
      g.computeVertexNormals();
      return g;
    };
    return { geo: make(RADIUS), halo: make(RADIUS * 2.2) };
  }, [id]);
  useFrame(({ clock }) => {                      // selected artery pulses
    if (mat.current) mat.current.emissiveIntensity = selected ? 0.9 + 0.5 * Math.sin(clock.elapsedTime * 4) : hover ? 0.55 : 0.25;
  });
  return (
    <group>
      <mesh geometry={geo}
        onClick={(e) => { e.stopPropagation(); onSelect(id); }}
        onPointerOver={(e) => { e.stopPropagation(); setHover(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHover(false); document.body.style.cursor = "auto"; }}>
        <meshStandardMaterial ref={mat} color={color} emissive={color} />
      </mesh>
      {selected && (
        <mesh geometry={halo} raycast={() => null}>
          <meshBasicMaterial color={color} transparent opacity={0.28} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

function Label({ id, selected, pct }) {
  const mid = PATHS[id][Math.floor(PATHS[id].length / 2)];
  const pos = mid.map((v, i) => (v - CENTER[i]) * SCALE);
  return (
    <Html position={pos} center distanceFactor={3.2} style={{ pointerEvents: "none" }}>
      <span className={"tag" + (selected ? " on" : "")}>{id}{pct != null ? ` ${pct}%` : ""}</span>
    </Html>
  );
}

function Model({ colors, pcts, selected, onSelect }) {
  const { scene } = useGLTF("/heart_coronary.glb");
  useEffect(() => {      // heart is display-only; clicks go to the artery tubes
    scene.traverse((o) => { if (o.isMesh) { o.raycast = () => {}; o.frustumCulled = false; } });
  }, [scene]);
  // Safety: measure the loaded heart and force it to 156 model-units tall and centred,
  // so it always matches the artery coordinates whatever transform the file carries.
  const fit = useMemo(() => {
    scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
    const k = 156 / size.y;
    console.log("heart size", size.toArray().map((v) => +v.toFixed(2)), "fit scale", +k.toFixed(4));
    return { k, pos: c.multiplyScalar(-k).toArray() };
  }, [scene]);
  return (
    <>
      <group scale={SCALE} position={CENTER.map((v) => -v * SCALE)}>
        <primitive object={scene} scale={fit.k} position={fit.pos} />
        {Object.keys(PATHS).map((id) => (
          <Vessel key={id} id={id} color={colors[id] || "#9ca3af"} selected={selected === id} onSelect={onSelect} />
        ))}
      </group>
      {Object.keys(PATHS).map((id) => <Label key={id} id={id} selected={selected === id} pct={pcts?.[id]} />)}
    </>
  );
}

class Boundary extends Component {
  state = { err: null };
  static getDerivedStateFromError(err) { return { err }; }
  render() {
    return this.state.err
      ? <div style={{ padding: 16, color: "#fca5a5" }}>3D viewer error: {String(this.state.err.message || this.state.err)}</div>
      : this.props.children;
  }
}

function HeartCanvas({ colors, pcts, selected, onSelect }) {
  const ctl = useRef();
  return (
    <>
    <div style={{ position: "absolute", top: 8, left: 8, zIndex: 2, fontSize: 11, color: "#cbd5e1", background: "#0b1220cc", padding: "6px 8px", borderRadius: 6 }}>
      <div style={{ marginBottom: 3 }}>Predicted stenosis probability</div>
      <div style={{ width: 150, height: 8, borderRadius: 4, background: "linear-gradient(90deg,#22c55e,#facc15,#ef4444)" }} />
      <div style={{ display: "flex", justifyContent: "space-between", width: 150, marginTop: 2 }}><span>0%</span><span>33%</span><span>66%</span><span>100%</span></div>
      <div style={{ display: "flex", justifyContent: "space-between", width: 150 }}><span>low</span><span>moderate</span><span>high</span></div>
    </div>
    <button onClick={() => ctl.current?.reset()}
      style={{ position: "absolute", top: 8, right: 8, zIndex: 2, background: "#1d4ed8", border: 0, borderRadius: 6, padding: "4px 10px" }}>
      Reset view
    </button>
    <Canvas camera={{ position: [0, 0.3, 6.5], fov: 40 }} onPointerMissed={() => onSelect("CAD")}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[3, 4, 5]} intensity={1.6} />
      <directionalLight position={[-3, -2, -4]} intensity={0.9} />
      <Suspense fallback={<Html center><span className="tag">Loading 3D model...</span></Html>}>
        <Model colors={colors} pcts={pcts} selected={selected} onSelect={onSelect} />
      </Suspense>
      <OrbitControls ref={ctl} enablePan={false} zoomSpeed={0.6} minDistance={3.6} maxDistance={10} />
    </Canvas>
    </>
  );
}

// `pcts` (optional): { LAD: 78, LCX: 67, RCA: 40 } adds the percentage to each label.
export default function Heart(p) { return <Boundary><HeartCanvas {...p} /></Boundary>; }
useGLTF.preload("/heart_coronary.glb");
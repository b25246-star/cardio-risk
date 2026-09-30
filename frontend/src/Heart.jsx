import { Component, Suspense, useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html, useGLTF } from "@react-three/drei";
import * as THREE from "three";

// Artery centre-lines in the GLB's raw units (x = patient's left, y = up, z = front),
// snapped onto the heart surface. Tweak these points to re-route an artery.
const PATHS = {
  LAD: [[0.6,10.6,2.31],[1.3,9.2,2.26],[2.0,7.6,1.92],[2.8,5.6,1.47],[3.4,3.6,2.3],[3.7,1.8,2.33],[3.6,0.6,2.38]],
  RCA: [[-0.4,10.4,2.34],[-1.6,10.0,2.35],[-2.8,8.6,2.27],[-3.3,6.6,1.94],[-3.0,4.4,2.3],[-2.0,2.6,2.35],[-0.6,1.4,2.35]],
  LCX: [[1.4,10.4,2.32],[2.8,10.0,2.35],[4.2,9.0,2.34],[4.3,7.5,-2.21],[3.6,6.0,-2.71],[2.4,5.2,-3.09],[1.0,5.0,-3.13]],
};
const RADIUS = 0.14;

function Vessel({ id, color, selected, onSelect }) {
  const [hover, setHover] = useState(false);
  const pts = useMemo(() => PATHS[id].map((p) => new THREE.Vector3(...p)), [id]);
  const geo = useMemo(() => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 80, RADIUS, 12), [pts]);
  const glow = selected ? 0.9 : hover ? 0.55 : 0.25;
  return (
    <group>
      <mesh geometry={geo}
        onClick={(e) => { e.stopPropagation(); onSelect(id); }}
        onPointerOver={(e) => { e.stopPropagation(); setHover(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHover(false); document.body.style.cursor = "auto"; }}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={glow} />
      </mesh>
      <Html position={pts[3]} center distanceFactor={3.2} style={{ pointerEvents: "none" }}>
        <span className={"tag" + (selected ? " on" : "")}>{id}</span>
      </Html>
    </group>
  );
}

function Model({ colors, selected, onSelect }) {
  const { scene } = useGLTF("/heart.glb");
  useEffect(() => {          // static rest pose; clicks go to arteries only
    scene.traverse((o) => { if (o.isMesh) { o.raycast = () => {}; o.frustumCulled = false; } });
  }, [scene]);
  // model is ~0.17 units tall (root scale 0.01): scale x12 and centre it
  return (
    <group scale={12} position={[0.0666, -1.027, 0.024]}>
      <primitive object={scene} />
      <group scale={0.01}>
        {Object.keys(PATHS).map((id) => (
          <Vessel key={id} id={id} color={colors[id] || "#9ca3af"} selected={selected === id} onSelect={onSelect} />
        ))}
      </group>
    </group>
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

function HeartCanvas({ colors, selected, onSelect }) {
  return (
    <Canvas camera={{ position: [0, 0.1, 5.6], fov: 45 }} onPointerMissed={() => onSelect("CAD")}>
      <ambientLight intensity={1.1} />
      <directionalLight position={[3, 4, 5]} intensity={1.6} />
      <directionalLight position={[-3, -2, -4]} intensity={0.8} />
      <Suspense fallback={<Html center><span className="tag">Loading 3D model…</span></Html>}>
        <Model colors={colors} selected={selected} onSelect={onSelect} />
      </Suspense>
      <OrbitControls enablePan={false} minDistance={2} maxDistance={9} />
    </Canvas>
  );
}
export default function Heart(p) { return <Boundary><HeartCanvas {...p} /></Boundary>; }
useGLTF.preload("/heart.glb");
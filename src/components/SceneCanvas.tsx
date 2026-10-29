import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  createDemoScene,
  createTitanScene,
  type SceneBundle,
  type SceneMode,
  type TitanSceneData,
} from "../lib/demoScene";

export interface SceneStats {
  pointCount: number;
  cameraCount: number;
  selected: string | null;
  fps: number;
}

interface SceneCanvasProps {
  mode: SceneMode;
  onStats: (stats: SceneStats) => void;
  scene?: TitanSceneData | null;
}

function disposeMaterial(material: THREE.Material | THREE.Material[]) {
  if (Array.isArray(material)) {
    material.forEach((item) => item.dispose());
  } else {
    material.dispose();
  }
}

export function SceneCanvas({ mode, onStats, scene }: SceneCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const demoRef = useRef<SceneBundle | null>(null);
  const modeRef = useRef(mode);
  const onStatsRef = useRef(onStats);
  const statsRef = useRef<SceneStats>({
    pointCount: 0,
    cameraCount: 0,
    selected: null,
    fps: 0,
  });

  modeRef.current = mode;
  onStatsRef.current = onStats;

  useEffect(() => {
    demoRef.current?.setMode(mode);
  }, [mode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const world = new THREE.Scene();
    world.background = new THREE.Color("#0B1116");
    world.fog = new THREE.Fog("#0B1116", 9, 18);

    const camera = new THREE.PerspectiveCamera(
      48,
      container.clientWidth / container.clientHeight,
      0.1,
      60,
    );
    camera.position.set(4.4, 2.7, 4.6);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.25, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.6;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI * 0.92;

    world.add(new THREE.AmbientLight("#E6F7F5", 0.9));
    const keyLight = new THREE.DirectionalLight("#FFFFFF", 2.6);
    keyLight.position.set(3, 6, 4);
    world.add(keyLight);
    const rimLight = new THREE.PointLight("#2ED3BE", 7, 16);
    rimLight.position.set(-4.5, 2, -3.5);
    world.add(rimLight);

    const bundle = scene ? createTitanScene(scene) : createDemoScene();
    bundle.setMode(modeRef.current);
    world.add(bundle.group);
    demoRef.current = bundle;
    statsRef.current = {
      pointCount: bundle.pointCount,
      cameraCount: bundle.cameraCount,
      selected: null,
      fps: 0,
    };
    onStatsRef.current({ ...statsRef.current });

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.16;
    const pointer = new THREE.Vector2();

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects([bundle.points, bundle.mesh], true);
      if (hits.length === 0) {
        bundle.selection.visible = false;
        if (statsRef.current.selected !== null) {
          statsRef.current.selected = null;
          onStatsRef.current({ ...statsRef.current });
        }
        return;
      }
      const hit = hits[0];
      bundle.selection.position.copy(hit.point);
      bundle.selection.visible = true;
      statsRef.current.selected = `${hit.point.x.toFixed(2)}, ${hit.point.y.toFixed(2)}, ${hit.point.z.toFixed(2)}`;
      onStatsRef.current({ ...statsRef.current });
    };
    renderer.domElement.addEventListener("click", handleClick);

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    let animationFrame = 0;
    let previous = performance.now();
    let elapsed = 0;
    let lastStatsAt = 0;

    const animate = (now: number) => {
      animationFrame = requestAnimationFrame(animate);
      const delta = Math.min((now - previous) / 1000, 0.1);
      previous = now;
      elapsed += delta;
      const instantFps = delta > 0 ? 1 / delta : 0;
      statsRef.current.fps =
        statsRef.current.fps === 0
          ? instantFps
          : statsRef.current.fps * 0.9 + instantFps * 0.1;
      bundle.update(elapsed);
      controls.update();
      renderer.render(world, camera);
      if (now - lastStatsAt >= 500) {
        lastStatsAt = now;
        onStatsRef.current({ ...statsRef.current });
      }
    };
    animationFrame = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      bundle.group.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          disposeMaterial(object.material);
        } else if (object instanceof THREE.Points) {
          object.geometry.dispose();
          disposeMaterial(object.material);
        } else if (object instanceof THREE.Line) {
          object.geometry.dispose();
          disposeMaterial(object.material);
        }
      });
      demoRef.current = null;
    };
  }, [scene]);

  return <div className="scene-canvas" ref={containerRef} aria-hidden="true" />;
}

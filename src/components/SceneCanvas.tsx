import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";
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

export interface SceneCanvasHandle {
  resetView(): void;
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

export const SceneCanvas = forwardRef<
  SceneCanvasHandle,
  SceneCanvasProps
>(function SceneCanvas({ mode, onStats, scene }, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const demoRef = useRef<SceneBundle | null>(null);
  const resetViewRef = useRef<() => void>(() => {});
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
  useImperativeHandle(
    ref,
    () => ({
      resetView() {
        resetViewRef.current();
      },
    }),
    [],
  );

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

    const camera = new THREE.PerspectiveCamera(
      48,
      container.clientWidth / container.clientHeight,
      0.01,
      120,
    );

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
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
    const center = bundle.sceneCenter;
    const radius = Math.max(bundle.sceneRadius, 1e-3);
    const fitDistance = Math.max(radius * 2.4, 1.2);
    const offset = new THREE.Vector3(0.68, 0.42, 0.68)
      .normalize()
      .multiplyScalar(fitDistance);
    camera.near = Math.max(radius * 0.002, 0.01);
    camera.far = Math.max(radius * 60, 20);
    camera.position.copy(center).add(offset);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.minDistance = Math.max(radius * 0.04, 0.05);
    controls.maxDistance = Math.max(radius * 12, 12);
    world.fog = new THREE.Fog(
      "#0B1116",
      radius * 4,
      radius * 9,
    );
    const homePosition = camera.position.clone();
    const homeTarget = controls.target.clone();
    resetViewRef.current = () => {
      camera.position.copy(homePosition);
      controls.target.copy(homeTarget);
      controls.update();
    };
    bundle.setViewCamera(camera);
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
        } else if (object instanceof THREE.Sprite) {
          disposeMaterial(object.material);
        }
      });
      demoRef.current = null;
    };
  }, [scene]);

  return <div className="scene-canvas" ref={containerRef} aria-hidden="true" />;
});

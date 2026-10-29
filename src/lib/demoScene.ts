import * as THREE from "three";

export type SceneMode = "camera" | "points" | "mesh";

export interface SceneBundle {
  group: THREE.Group;
  points: THREE.Points;
  mesh: THREE.Group;
  cameraPath: THREE.Group;
  selection: THREE.Mesh;
  pointCount: number;
  cameraCount: number;
  setMode(mode: SceneMode): void;
  update(elapsed: number): void;
}

export interface TitanCamera {
  id?: number;
  position: number[];
  forward: number[];
  width?: number;
  height?: number;
}

export interface TitanSceneData {
  version?: string;
  engine?: string;
  taskId?: string;
  projectDir?: string;
  points: number[][];
  cameras: TitanCamera[];
  renders?: { name: string; path: string; view: string }[];
  novelView?: string;
  metrics?: Record<string, number | string>;
  history?: Record<string, unknown>[];
  config?: Record<string, unknown>;
}

const ACCENT = "#2ED3BE";
const GOLD = "#D9A84C";
const MESH_COLOR = "#9AAFB8";

function buildPointCloud(count: number): THREE.Points {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const teal = new THREE.Color(ACCENT);
  const gold = new THREE.Color(GOLD);
  const tmp = new THREE.Color();

  for (let i = 0; i < count; i += 1) {
    const theta = Math.acos(2 * Math.random() - 1);
    const phi = Math.random() * Math.PI * 2;
    const band =
      0.24 * Math.sin(theta * 5 + phi * 2) +
      0.16 * Math.sin(theta * 9 - phi * 3) +
      0.1 * Math.pow(Math.sin(theta * 7) * Math.sin(phi * 4), 3);
    const radius = 2.05 + band;

    positions[i * 3] = radius * Math.sin(theta) * Math.cos(phi);
    positions[i * 3 + 1] = radius * Math.cos(theta) * 0.92;
    positions[i * 3 + 2] = radius * Math.sin(theta) * Math.sin(phi);

    const t = THREE.MathUtils.clamp((radius - 1.65) / 1.15, 0, 1);
    tmp.copy(teal).lerp(gold, t);
    colors[i * 3] = tmp.r;
    colors[i * 3 + 1] = tmp.g;
    colors[i * 3 + 2] = tmp.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.045,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}

function buildMesh(): THREE.Group {
  const geometry = new THREE.IcosahedronGeometry(2.15, 3);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const vertex = new THREE.Vector3();

  for (let i = 0; i < position.count; i += 1) {
    vertex.fromBufferAttribute(position, i).normalize();
    const wobble =
      0.12 * Math.sin(vertex.x * 4 + vertex.y * 3) +
      0.09 * Math.sin(vertex.z * 5) +
      0.06 * Math.sin(vertex.x * 7 - vertex.z * 4);
    vertex.multiplyScalar(1 + wobble);
    position.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();

  const group = new THREE.Group();
  const solid = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: MESH_COLOR,
      roughness: 0.68,
      metalness: 0.08,
      flatShading: true,
    }),
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.34,
    }),
  );
  group.add(solid, edges);
  return group;
}

function buildCameraPath(frames: number): {
  path: THREE.Group;
  curve: THREE.CatmullRomCurve3;
  scanner: THREE.Mesh;
} {
  const keyPoints: THREE.Vector3[] = [];
  for (let i = 0; i < frames; i += 1) {
    const t = i / frames;
    const angle = t * Math.PI * 2;
    const radius = 3.1 + 0.55 * Math.sin(angle * 3 + t * 4);
    const y = 0.35 + 1.35 * Math.sin(angle * 1.7);
    keyPoints.push(
      new THREE.Vector3(
        Math.cos(angle) * radius,
        y,
        Math.sin(angle) * radius,
      ),
    );
  }

  const curve = new THREE.CatmullRomCurve3(keyPoints, true);
  const path = new THREE.Group();
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(curve.getPoints(180)),
    new THREE.LineBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.8,
    }),
  );
  path.add(line);

  const markerMaterial = new THREE.MeshBasicMaterial({
    color: "#9FE8DE",
    transparent: true,
    opacity: 0.85,
  });
  const markerGeometry = new THREE.BoxGeometry(0.13, 0.13, 0.2);
  for (const point of keyPoints) {
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.copy(point);
    path.add(marker);
  }

  const scanner = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 14, 14),
    new THREE.MeshBasicMaterial({ color: GOLD }),
  );
  scanner.visible = false;
  path.add(scanner);

  return { path, curve, scanner };
}

function buildTitanPoints(entries: number[][]): THREE.Points {
  const positions = new Float32Array(entries.length * 3);
  const colors = new Float32Array(entries.length * 3);
  const teal = new THREE.Color(ACCENT);
  const gold = new THREE.Color(GOLD);
  const tmp = new THREE.Color();

  entries.forEach((entry, index) => {
    positions[index * 3] = entry[0];
    positions[index * 3 + 1] = entry[1];
    positions[index * 3 + 2] = entry[2];
    const r = entry[3];
    const g = entry[4];
    const b = entry[5];
    if (r != null && g != null && b != null) {
      colors[index * 3] = r / 255;
      colors[index * 3 + 1] = g / 255;
      colors[index * 3 + 2] = b / 255;
    } else {
      const radius = Math.hypot(entry[0], entry[1], entry[2]);
      const t = THREE.MathUtils.clamp((radius - 1.65) / 1.15, 0, 1);
      tmp.copy(teal).lerp(gold, t);
      colors[index * 3] = tmp.r;
      colors[index * 3 + 1] = tmp.g;
      colors[index * 3 + 2] = tmp.b;
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0.94,
    sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}

function buildTitanCameraPath(cameras: TitanCamera[]): {
  path: THREE.Group;
  curve: THREE.CatmullRomCurve3;
  scanner: THREE.Mesh;
} {
  const keyPoints = cameras
    .slice()
    .sort((a, b) => (a.id ?? 0) - (b.id ?? 0))
    .map(
      (camera) =>
        new THREE.Vector3(camera.position[0], camera.position[1], camera.position[2]),
    );
  const path = new THREE.Group();
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(keyPoints),
    new THREE.LineBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.82,
    }),
  );
  path.add(line);

  const markerMaterial = new THREE.MeshBasicMaterial({
    color: "#9FE8DE",
    transparent: true,
    opacity: 0.85,
  });
  const markerGeometry = new THREE.BoxGeometry(0.12, 0.12, 0.2);
  for (const point of keyPoints) {
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.copy(point);
    path.add(marker);
  }

  const scanner = new THREE.Mesh(
    new THREE.SphereGeometry(0.11, 14, 14),
    new THREE.MeshBasicMaterial({ color: GOLD }),
  );
  scanner.visible = false;
  path.add(scanner);

  const curvePoints =
    keyPoints.length >= 2
      ? keyPoints
      : [new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0)];
  const closed = keyPoints.length > 2;
  const curve = new THREE.CatmullRomCurve3(curvePoints, closed, "centripetal", 0.5);
  return { path, curve, scanner };
}

function buildGround(): THREE.Group {
  const group = new THREE.Group();
  const grid = new THREE.GridHelper(16, 18, ACCENT, "#5F6C73");
  const gridMaterials = Array.isArray(grid.material)
    ? grid.material
    : [grid.material];
  gridMaterials.forEach((material) => {
    material.transparent = true;
    material.opacity = 0.18;
  });
  group.add(grid);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(2.75, 2.9, 72),
    new THREE.MeshBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -2.35;
  group.add(ring);
  return group;
}

export function createDemoScene(): SceneBundle {
  const group = new THREE.Group();
  const points = buildPointCloud(48_000);
  const mesh = buildMesh();
  const { path: cameraPath, curve, scanner } = buildCameraPath(24);
  const ground = buildGround();
  const selection = new THREE.Mesh(
    new THREE.SphereGeometry(0.17, 18, 18),
    new THREE.MeshBasicMaterial({
      color: "#FFD37A",
      transparent: true,
      opacity: 0.95,
    }),
  );
  selection.visible = false;

  group.add(points, mesh, cameraPath, ground, selection);

  const demo: SceneBundle = {
    group,
    points,
    mesh,
    cameraPath,
    selection,
    pointCount: points.geometry.attributes.position.count,
    cameraCount: 24,
    setMode(mode) {
      points.visible = mode === "points";
      mesh.visible = mode === "mesh";
      cameraPath.visible = mode === "camera";
      scanner.visible = mode === "camera";
      selection.visible = false;
    },
    update(elapsed) {
      if (cameraPath.visible) {
        scanner.position.copy(curve.getPointAt((elapsed * 0.06) % 1));
      }
      if (selection.visible) {
        const pulse = 1 + 0.14 * Math.sin(elapsed * 5);
        selection.scale.set(pulse, pulse, pulse);
      }
    },
  };

  demo.setMode("points");
  return demo;
}

export function createTitanScene(data: TitanSceneData): SceneBundle {
  const group = new THREE.Group();
  const points = buildTitanPoints(data.points);
  const mesh = new THREE.Group();
  const { path: cameraPath, curve, scanner } = buildTitanCameraPath(data.cameras);
  const ground = buildGround();
  const selection = new THREE.Mesh(
    new THREE.SphereGeometry(0.17, 18, 18),
    new THREE.MeshBasicMaterial({
      color: "#FFD37A",
      transparent: true,
      opacity: 0.95,
    }),
  );
  selection.visible = false;

  group.add(points, mesh, cameraPath, ground, selection);

  const scene: SceneBundle = {
    group,
    points,
    mesh,
    cameraPath,
    selection,
    pointCount: data.points.length,
    cameraCount: data.cameras.length,
    setMode(mode) {
      points.visible = mode === "points";
      mesh.visible = mode === "mesh";
      cameraPath.visible = mode === "camera";
      scanner.visible = mode === "camera";
      selection.visible = false;
    },
    update(elapsed) {
      if (cameraPath.visible && data.cameras.length > 1) {
        scanner.position.copy(curve.getPointAt((elapsed * 0.05) % 1));
      }
      if (selection.visible) {
        const pulse = 1 + 0.14 * Math.sin(elapsed * 5);
        selection.scale.set(pulse, pulse, pulse);
      }
    },
  };

  scene.setMode("points");
  return scene;
}

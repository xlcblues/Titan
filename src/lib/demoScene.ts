import * as THREE from "three";

export type SceneMode = "camera" | "points" | "mesh";

export interface SceneBundle {
  group: THREE.Group;
  points: THREE.Points;
  mesh: THREE.Group;
  cameraPath: THREE.Group;
  markers: THREE.Sprite[];
  selection: THREE.Mesh;
  pointCount: number;
  cameraCount: number;
  sceneCenter: THREE.Vector3;
  sceneRadius: number;
  pointCenter: THREE.Vector3;
  pointRadius: number;
  cameraCenter: THREE.Vector3;
  cameraRadius: number;
  viewDirection: THREE.Vector3;
  setMode(mode: SceneMode): void;
  setViewCamera(camera: THREE.PerspectiveCamera): void;
  update(elapsed: number): void;
}

export interface TitanCamera {
  id?: number;
  path?: string;
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
  poseSource?: string;
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

function makeCircleTexture(): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  if (!context) {
    return new THREE.Texture(canvas);
  }
  const gradient = context.createRadialGradient(32, 32, 2, 32, 32, 30);
  gradient.addColorStop(0, "rgba(214, 250, 245, 0.95)");
  gradient.addColorStop(0.6, "rgba(46, 211, 190, 0.62)");
  gradient.addColorStop(1, "rgba(46, 211, 190, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function makeLabelTexture(text: string): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 48;
  const context = canvas.getContext("2d");
  if (!context) {
    return new THREE.Texture(canvas);
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = "600 28px 'Segoe UI', 'Microsoft YaHei UI', sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.lineWidth = 5;
  context.strokeStyle = "rgba(11, 17, 22, 0.9)";
  context.strokeText(text, 64, 24);
  context.fillStyle = "#F1F5F4";
  context.fillText(text, 64, 24);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function buildPointCloud(count: number, radius: number): THREE.Points {
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
    size: Math.max(radius * 0.02, 0.03),
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

function buildCameraPath(
  frames: number,
  radius: number,
): {
  path: THREE.Group;
  curve: THREE.CatmullRomCurve3;
  scanner: THREE.Mesh;
  markers: THREE.Sprite[];
} {
  const circleTexture = makeCircleTexture();
  const markerSize = Math.max(radius * 0.06, 0.16);
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
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(
      curve,
      Math.max(96, frames * 10),
      Math.max(radius * 0.006, 0.015),
      8,
      true,
    ),
    new THREE.MeshBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
    }),
  );
  path.add(tube);

  const coneGeometry = new THREE.ConeGeometry(
    markerSize * 0.58,
    markerSize * 1.5,
    5,
  );
  const coneMaterial = new THREE.MeshBasicMaterial({
    color: GOLD,
    transparent: true,
    opacity: 0.98,
  });
  const markerSpriteMaterial = new THREE.SpriteMaterial({
    map: circleTexture,
    color: "#9FE8DE",
    transparent: true,
    opacity: 0.9,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: false,
  });
  const markers: THREE.Sprite[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const orientation = new THREE.Quaternion();
  for (const [index, point] of keyPoints.entries()) {
    const cone = new THREE.Mesh(coneGeometry, coneMaterial);
    cone.position.copy(point);
    const previous = keyPoints[(keyPoints.indexOf(point) + frames - 1) % frames];
    const next = keyPoints[(keyPoints.indexOf(point) + 1) % frames];
    const forward = next
      .clone()
      .sub(previous)
      .normalize();
    orientation.setFromUnitVectors(up, forward);
    cone.quaternion.copy(orientation);
    path.add(cone);
    const marker = new THREE.Sprite(markerSpriteMaterial);
    marker.position.copy(point);
    marker.scale.set(22, 22, 1);
    markers.push(marker);
    path.add(marker);

    const label = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: makeLabelTexture(String(index + 1)),
        transparent: true,
        depthTest: false,
        depthWrite: false,
        sizeAttenuation: false,
      }),
    );
    label.position.copy(point).addScaledVector(up, markerSize * 1.5);
    label.scale.set(64, 24, 1);
    markers.push(label);
    path.add(label);
  }

  const scanner = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 14, 14),
    new THREE.MeshBasicMaterial({ color: GOLD }),
  );
  scanner.visible = false;
  path.add(scanner);

  return { path, curve, scanner, markers };
}

function buildTitanPoints(entries: number[][], radius: number): THREE.Points {
  const positions = new Float32Array(entries.length * 3);
  const colors = new Float32Array(entries.length * 3);
  const teal = new THREE.Color(ACCENT);
  const gold = new THREE.Color(GOLD);
  const tmp = new THREE.Color();

  let minPointRadius = Infinity;
  let maxPointRadius = -Infinity;
  for (const entry of entries) {
    const pointRadius = Math.hypot(entry[0], entry[1], entry[2]);
    minPointRadius = Math.min(minPointRadius, pointRadius);
    maxPointRadius = Math.max(maxPointRadius, pointRadius);
  }
  const pointRadiusSpan = maxPointRadius - minPointRadius;

  entries.forEach((entry, index) => {
    positions[index * 3] = entry[0];
    positions[index * 3 + 1] = entry[1];
    positions[index * 3 + 2] = entry[2];
    const r = entry[3];
    const g = entry[4];
    const b = entry[5];
    const hasColor = r != null && g != null && b != null;
    const channelSum = hasColor ? r + g + b : 0;
    if (hasColor && channelSum >= 18) {
      colors[index * 3] = r / 255;
      colors[index * 3 + 1] = g / 255;
      colors[index * 3 + 2] = b / 255;
    } else {
      const pointRadius = Math.hypot(entry[0], entry[1], entry[2]);
      const t =
        pointRadiusSpan > 1e-6
          ? THREE.MathUtils.clamp(
              (pointRadius - minPointRadius) / pointRadiusSpan,
              0,
              1,
            )
          : 0.5;
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
    size: Math.max(radius * 0.022, 0.08),
    vertexColors: true,
    transparent: true,
    opacity: 0.96,
    sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}

function buildTitanCameraPath(
  cameras: TitanCamera[],
  cameraRadius: number,
): {
  path: THREE.Group;
  curve: THREE.CatmullRomCurve3;
  scanner: THREE.Mesh;
  markers: THREE.Sprite[];
} {
  const orderOf = (
    camera: TitanCamera,
    index: number,
  ): { order: number; label: string } => {
    const basename = (camera.path ?? "").split(/[\\/]/).pop() ?? "";
    const digits = basename.match(/(\d+)/);
    if (digits) {
      return { order: Number(digits[0]), label: digits[0] };
    }
    return { order: camera.id ?? index, label: String(camera.id ?? index) };
  };
  const sortedCameras = cameras
    .map((camera, index) => ({ camera, ...orderOf(camera, index) }))
    .sort((a, b) => a.order - b.order);
  const keyPoints = sortedCameras.map(
    ({ camera }) =>
      new THREE.Vector3(
        camera.position[0],
        camera.position[1],
        camera.position[2],
      ),
  );
  const path = new THREE.Group();
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(keyPoints),
    new THREE.LineBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.95,
    }),
  );
  path.add(line);

  const circleTexture = makeCircleTexture();
  const markerSize = Math.max(cameraRadius * 0.3, 0.12);
  const coneGeometry = new THREE.ConeGeometry(
    markerSize * 0.6,
    markerSize * 1.6,
    5,
  );
  const coneMaterial = new THREE.MeshBasicMaterial({
    color: GOLD,
    transparent: true,
    opacity: 0.98,
  });
  const frustumGeometry = new THREE.ConeGeometry(
    markerSize * 0.95,
    markerSize * 2.2,
    4,
  );
  const frustumMaterial = new THREE.MeshBasicMaterial({
    color: ACCENT,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  const markerSpriteMaterial = new THREE.SpriteMaterial({
    map: circleTexture,
    color: "#9FE8DE",
    transparent: true,
    opacity: 0.95,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const markers: THREE.Sprite[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const orientation = new THREE.Quaternion();
  const showLabels = cameras.length <= 80;
  for (const [index, point] of keyPoints.entries()) {
    const { camera, label: labelText } = sortedCameras[index];
    const cone = new THREE.Mesh(coneGeometry, coneMaterial);
    cone.position.copy(point);
    const frustum = new THREE.Mesh(frustumGeometry, frustumMaterial);
    frustum.position.copy(point);
    const forward = new THREE.Vector3(
      camera.forward[0],
      camera.forward[1],
      camera.forward[2],
    ).normalize();
    orientation.setFromUnitVectors(up, forward);
    cone.quaternion.copy(orientation);
    frustum.quaternion.copy(orientation);
    path.add(cone);
    path.add(frustum);
    const marker = new THREE.Sprite(markerSpriteMaterial);
    marker.position.copy(point);
    marker.scale.set(markerSize * 0.8, markerSize * 0.8, 1);
    markers.push(marker);
    path.add(marker);

    if (showLabels) {
      const label = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: makeLabelTexture(labelText),
          transparent: true,
          depthTest: false,
          depthWrite: false,
          sizeAttenuation: true,
        }),
      );
      label.position.copy(point).addScaledVector(up, markerSize * 2.0);
      label.scale.set(markerSize * 1.5, markerSize * 0.56, 1);
      markers.push(label);
      path.add(label);
    }
  }

  const scanner = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(markerSize * 0.55, 0.14), 14, 14),
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
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(
      curve,
      Math.max(96, keyPoints.length * 12),
      Math.max(cameraRadius * 0.03, 0.008),
      8,
      closed,
    ),
    new THREE.MeshBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
    }),
  );
  path.add(tube);
  return { path, curve, scanner, markers };
}

function buildGround(radius: number): THREE.Group {
  const group = new THREE.Group();
  const gridSize = Math.max(radius * 3, 4);
  const grid = new THREE.GridHelper(
    gridSize,
    Math.min(32, Math.max(12, Math.round(gridSize / 0.9))),
    ACCENT,
    "#5F6C73",
  );
  const gridMaterials = Array.isArray(grid.material)
    ? grid.material
    : [grid.material];
  gridMaterials.forEach((material) => {
    material.transparent = true;
    material.opacity = 0.18;
  });
  group.add(grid);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(radius * 0.58, radius * 0.61, 72),
    new THREE.MeshBasicMaterial({
      color: ACCENT,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -radius * 0.48;
  group.add(ring);
  return group;
}

export function createDemoScene(): SceneBundle {
  const group = new THREE.Group();
  const sceneCenter = new THREE.Vector3(0, 0.25, 0);
  const sceneRadius = 5.2;
  const points = buildPointCloud(48_000, sceneRadius);
  const mesh = buildMesh();
  const {
    path: cameraPath,
    curve,
    scanner,
    markers,
  } = buildCameraPath(24, sceneRadius);
  const ground = buildGround(sceneRadius);
  ground.position.y = sceneCenter.y - sceneRadius;
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

  let viewCamera: THREE.PerspectiveCamera | null = null;
  const demo: SceneBundle = {
    group,
    points,
    mesh,
    cameraPath,
    markers,
    selection,
    pointCount: points.geometry.attributes.position.count,
    cameraCount: 24,
    sceneCenter,
    sceneRadius,
    pointCenter: sceneCenter.clone(),
    pointRadius: sceneRadius,
    cameraCenter: sceneCenter.clone(),
    cameraRadius: 4,
    viewDirection: new THREE.Vector3(0, 0, 1),
    setMode(mode) {
      const pointMaterial = points.material as THREE.PointsMaterial;
      const isCameraMode = mode === "camera";
      points.visible = mode === "points" || isCameraMode;
      pointMaterial.opacity = isCameraMode ? 0.3 : 0.96;
      mesh.visible = mode === "mesh";
      cameraPath.visible = isCameraMode;
      scanner.visible = isCameraMode;
      selection.visible = false;
    },
    setViewCamera(camera) {
      viewCamera = camera;
    },
    update(elapsed) {
      if (cameraPath.visible) {
        scanner.position.copy(curve.getPointAt((elapsed * 0.06) % 1));
      }
      const currentViewCamera = viewCamera;
      if (currentViewCamera) {
        markers.forEach((marker) => marker.lookAt(currentViewCamera.position));
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

function normalizeViewerY(scene: TitanSceneData): TitanSceneData {
  const flipY = scene.poseSource === "colmap" || scene.poseSource === "llff";
  if (!flipY) return scene;
  const flip = (values: number[]) => [values[0], -values[1], values[2]];
  return {
    ...scene,
    points: scene.points.map((point) => [
      point[0],
      -point[1],
      point[2],
      point[3],
      point[4],
      point[5],
    ]),
    cameras: scene.cameras.map((camera) => ({
      ...camera,
      position: flip(camera.position),
      forward: flip(camera.forward),
    })),
  };
}

export function createTitanScene(data: TitanSceneData): SceneBundle {
  const group = new THREE.Group();
  const viewerData = normalizeViewerY(data);
  const pointEntries = viewerData.points;
  const cameraEntries = viewerData.cameras;
  const pointBox = new THREE.Box3();
  const cameraBox = new THREE.Box3();
  pointEntries.forEach((point) => {
    pointBox.expandByPoint(new THREE.Vector3(point[0], point[1], point[2]));
  });
  cameraEntries.forEach((camera) => {
    cameraBox.expandByPoint(
      new THREE.Vector3(
        camera.position[0],
        camera.position[1],
        camera.position[2],
      ),
    );
  });

  const combinedBox = new THREE.Box3();
  if (!pointBox.isEmpty()) combinedBox.union(pointBox);
  if (!cameraBox.isEmpty()) combinedBox.union(cameraBox);
  const combinedCenter = combinedBox.getCenter(new THREE.Vector3());
  const combinedSize = combinedBox.getSize(new THREE.Vector3());
  const sceneRadius = Math.max(
    Math.max(combinedSize.x, combinedSize.y, combinedSize.z) / 2,
    1e-3,
  );

  const pointCenter = pointBox.isEmpty()
    ? new THREE.Vector3()
    : pointBox
        .getCenter(new THREE.Vector3())
        .sub(combinedCenter);
  const pointSize = pointBox.getSize(new THREE.Vector3());
  const pointRadius = pointBox.isEmpty()
    ? 1
    : Math.max(
        Math.max(pointSize.x, pointSize.y, pointSize.z) / 2,
        1e-3,
      );
  const cameraCenter = cameraBox.isEmpty()
    ? new THREE.Vector3()
    : cameraBox
        .getCenter(new THREE.Vector3())
        .sub(combinedCenter);
  const cameraSize = cameraBox.getSize(new THREE.Vector3());
  const cameraRadius = cameraBox.isEmpty()
    ? 1
    : Math.max(
        Math.max(cameraSize.x, cameraSize.y, cameraSize.z) / 2,
        1e-3,
      );
  const viewDirection = new THREE.Vector3();
  if (!pointBox.isEmpty() && !cameraBox.isEmpty()) {
    const pointMean = new THREE.Vector3();
    pointEntries.forEach((point) => {
      pointMean.x += point[0];
      pointMean.y += point[1];
      pointMean.z += point[2];
    });
    pointMean.divideScalar(pointEntries.length).sub(combinedCenter);
    viewDirection.copy(pointMean).sub(cameraCenter);
  }
  if (viewDirection.lengthSq() < 1e-6) {
    viewDirection.set(0, 0, 1);
  } else {
    viewDirection.normalize();
  }

  const points = buildTitanPoints(pointEntries, Math.max(pointRadius, 1e-3));
  const mesh = new THREE.Group();
  const {
    path: cameraPath,
    curve,
    scanner,
    markers,
  } = buildTitanCameraPath(cameraEntries, cameraRadius);
  const ground = buildGround(sceneRadius);
  ground.position.y = combinedCenter.y - sceneRadius;
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
  group.position.copy(combinedCenter).multiplyScalar(-1);

  let viewCamera: THREE.PerspectiveCamera | null = null;
  const scene: SceneBundle = {
    group,
    points,
    mesh,
    cameraPath,
    markers,
    selection,
    pointCount: pointEntries.length,
    cameraCount: cameraEntries.length,
    sceneCenter: new THREE.Vector3(),
    sceneRadius,
    pointCenter,
    pointRadius,
    cameraCenter,
    cameraRadius,
    viewDirection,
    setMode(mode) {
      const pointMaterial = points.material as THREE.PointsMaterial;
      const isCameraMode = mode === "camera";
      points.visible = mode === "points" || isCameraMode;
      pointMaterial.opacity = isCameraMode ? 0.3 : 0.96;
      mesh.visible = mode === "mesh";
      cameraPath.visible = isCameraMode;
      scanner.visible = isCameraMode;
      selection.visible = false;
    },
    setViewCamera(camera) {
      viewCamera = camera;
    },
    update(elapsed) {
      if (cameraPath.visible && cameraEntries.length > 1) {
        scanner.position.copy(curve.getPointAt((elapsed * 0.05) % 1));
      }
      const currentViewCamera = viewCamera;
      if (currentViewCamera) {
        markers.forEach((marker) => marker.lookAt(currentViewCamera.position));
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

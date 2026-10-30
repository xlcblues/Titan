const fs = require("fs");
const http = require("http");

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => resolve(JSON.parse(body)));
    }).on("error", reject);
  });
}

async function main() {
  const targets = await getJson("http://localhost:8315/json/list");
  const target = targets.find((item) => item.type === "page");
  if (!target) {
    throw new Error("No CDP page target found");
  }

  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let nextId = 1;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
    }
  });

  function send(method, params = {}) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }

  await send("Page.enable");
  await send("Runtime.enable");

  const projectId =
    process.argv[2] || "c7c10847-4388-4853-9729-d8c690486697";
  const sceneUrl = `http://localhost:1420/#/scene?project=${projectId}`;
  const location = await send("Runtime.evaluate", {
    expression: "location.href",
    returnByValue: true,
  });
  if (!location.result.value.includes("/scene")) {
    await send("Page.navigate", { url: sceneUrl });
    for (let attempt = 0; attempt < 40; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const ready = await send("Runtime.evaluate", {
        expression: "!!document.querySelector('.scene-stage')",
        returnByValue: true,
      });
      if (ready.result.value) break;
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 4000));

  const expression = `
    (async () => {
      await document.fonts.ready;
      const rect = (el) => el ? (() => {
        const r = el.getBoundingClientRect();
        return { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height), right: Math.round(r.right), bottom: Math.round(r.bottom) };
      })() : null;
      const canvas = document.querySelector('.scene-canvas canvas');
      const stage = document.querySelector('.scene-stage');
      const renderPanel = document.querySelector('.scene-panel--render');
      const renderBody = renderPanel ? renderPanel.querySelector('.panel__body') : null;
      const stageRect = rect(stage);
      const renderRect = rect(renderPanel);
      const overlap = stageRect && renderRect
        ? Math.max(0, Math.min(stageRect.right, renderRect.right) - Math.max(stageRect.left, renderRect.left)) > 0
          && Math.max(0, Math.min(stageRect.bottom, renderRect.bottom) - Math.max(stageRect.top, renderRect.top)) > 0
        : null;
      const info = {
        url: location.href,
        stats: (document.querySelector('.scene-hud') || { textContent: '' }).textContent.trim(),
        stage: stageRect,
        side: rect(document.querySelector('.scene-side')),
        render: renderRect,
        renderBody: rect(renderBody),
        renderBodyScrolls: renderBody ? { clientHeight: renderBody.clientHeight, scrollHeight: renderBody.scrollHeight } : null,
        renderImages: Array.from(document.querySelectorAll('.render-thumb__frame img')).slice(0, 6).map((img) => ({
          loaded: img.complete && img.naturalWidth > 0,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight
        })),
        overlapsStage: overlap
      };
      if (canvas) {
        const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
        if (gl) {
          const width = gl.drawingBufferWidth;
          const height = gl.drawingBufferHeight;
          const buffer = new Uint8Array(width * height * 4);
          gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
          const bg = [11, 17, 22];
          let lit = 0;
          let minX = 1e9;
          let minY = 1e9;
          let maxX = -1;
          let maxY = -1;
          let accentPixels = 0;
          let goldPixels = 0;
          let brightPixels = 0;
          for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
              const index = (y * width + x) * 4;
              const r = buffer[index];
              const g = buffer[index + 1];
              const b = buffer[index + 2];
              const dr = buffer[index] - bg[0];
              const dg = buffer[index + 1] - bg[1];
              const db = buffer[index + 2] - bg[2];
              if (dr * dr + dg * dg + db * db > 144) {
                lit += 1;
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
              }
              const accentDr = r - 46;
              const accentDg = g - 211;
              const accentDb = b - 190;
              if (accentDr * accentDr + accentDg * accentDg + accentDb * accentDb < 2500) {
                accentPixels += 1;
              }
              const goldDr = r - 217;
              const goldDg = g - 168;
              const goldDb = b - 76;
              if (goldDr * goldDr + goldDg * goldDg + goldDb * goldDb < 2500) {
                goldPixels += 1;
              }
              if (Math.max(r, g, b) > 130) {
                brightPixels += 1;
              }
            }
          }
          info.canvas = {
            width,
            height,
            litPixels: lit,
            coveragePercent: ((lit / (width * height)) * 1000) / 10,
            bbox: [minX, minY, maxX, maxY],
            accentPixels,
            goldPixels,
            brightPixels
          };
        }
      }
      return info;
    })()
  `;

  const evaluated = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true
  });
  const screenshot = await send("Page.captureScreenshot", { format: "png" });
  const output = require("path").join(process.cwd(), ".tmp_scene_ui.png");
  fs.writeFileSync(output, Buffer.from(screenshot.data, "base64"));

  await send("Runtime.evaluate", {
    expression: `document.querySelectorAll('.scene-toolbar .segmented button')[0].click(); true`
  });
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const cameraScreenshot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(
    require("path").join(process.cwd(), ".tmp_scene_camera.png"),
    Buffer.from(cameraScreenshot.data, "base64")
  );
  await send("Runtime.evaluate", {
    expression: `document.querySelectorAll('.scene-toolbar .segmented button')[1].click(); true`
  });

  console.log(JSON.stringify(evaluated.result.value, null, 2));
  console.log("screenshot:", output);
  socket.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

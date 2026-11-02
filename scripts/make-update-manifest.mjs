import { execSync } from "node:child_process";
import {
  existsSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(".");
const args = process.argv.slice(2);

function readArg(name, fallback) {
  const index = args.indexOf(name);
  if (index < 0 || !args[index + 1]) return fallback;
  return args[index + 1];
}

function fail(message) {
  console.error(message);
  process.exit(1);
}

const configPath = join(root, "src-tauri", "tauri.conf.json");
if (!existsSync(configPath)) {
  fail("Run this script from the Titan repository root.");
}

const config = JSON.parse(readFileSync(configPath, "utf8"));
const version = config.version;
const productName = config.productName;
const nsisDir = join(
  root,
  "src-tauri",
  "target",
  "release",
  "bundle",
  "nsis",
);

if (!existsSync(nsisDir)) {
  fail("NSIS bundle not found. Run `npm run tauri build` first.");
}

const candidates = readdirSync(nsisDir).filter(
  (name) =>
    name.endsWith(".exe") &&
    name.includes("setup") &&
    !name.toLowerCase().includes("uninstall"),
);
if (candidates.length === 0) {
  fail("No NSIS setup executable found in " + nsisDir);
}

const fileName =
  candidates.find((name) => name.includes(`${productName}_${version}`)) ??
  candidates[0];
const filePath = join(nsisDir, fileName);

let keyPath = process.env.TAURI_SIGNING_PRIVATE_KEY_PATH;
if (!keyPath) {
  const repoKey = join(root, "src-tauri", ".updater", "titan.key");
  const homeKey = join(homedir(), ".tauri", "titan.key");
  if (existsSync(repoKey)) keyPath = repoKey;
  else if (existsSync(homeKey)) keyPath = homeKey;
}
if (!keyPath || !existsSync(keyPath)) {
  fail(
    "Updater signing key not found. Set TAURI_SIGNING_PRIVATE_KEY_PATH or " +
      "generate a key with `npx tauri signer generate -w src-tauri\\.updater\\titan.key`.",
  );
}

const password = process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD;
if (!password) {
  fail(
    "Set TAURI_SIGNING_PRIVATE_KEY_PASSWORD before signing " +
      "the update manifest.",
  );
}

const signCommand = [
  "npx tauri signer sign",
  `--private-key-path ${JSON.stringify(keyPath)}`,
  `--app-version ${JSON.stringify(version)}`,
  JSON.stringify(filePath),
].join(" ");

let output;
try {
  output = execSync(signCommand, {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password },
  });
} catch (caught) {
  fail(caught.stderr || caught.message || "Failed to sign the installer.");
}
const signatureMatch = output.match(/Public signature:\s*(\S+)/);
if (!signatureMatch) {
  fail("Could not read the signature from the signer output.");
}

const releaseTag = `v${version}`;
const urlBase = readArg(
  "--url-base",
  "https://github.com/xlcblues/Titan/releases/download",
);
const notes = readArg("--notes", "Titan update");
const manifest = {
  version,
  notes,
  pub_date: new Date().toISOString(),
  platforms: {
    "windows-x86_64": {
      signature: signatureMatch[1],
      url: `${urlBase}/${releaseTag}/${fileName}`,
    },
  },
};

const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
const bundleOutPath = join(
  root,
  "src-tauri",
  "target",
  "release",
  "bundle",
  "latest.json",
);
const repoOutPath = join(root, "update", "latest.json");
writeFileSync(bundleOutPath, manifestText);
writeFileSync(repoOutPath, manifestText);

console.log(`Update manifest written to ${bundleOutPath}`);
console.log(`Synced to ${repoOutPath} for the GitHub release asset.`);
console.log("Upload these files as assets on the GitHub release:");
console.log(`  - ${filePath}`);
console.log(`  - ${repoOutPath} (must be named latest.json)`);

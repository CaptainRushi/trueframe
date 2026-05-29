import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const publicDir = path.join(__dirname, '../public');
const versionFilePath = path.join(publicDir, 'version.json');

// Ensure public directory exists
if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
}

const versionData = {
    version: new Date().getTime().toString(),
    buildDate: new Date().toISOString(),
};

fs.writeFileSync(versionFilePath, JSON.stringify(versionData, null, 2));

console.log(`Generated version.json at ${versionFilePath}`);

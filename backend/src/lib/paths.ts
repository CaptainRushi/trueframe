import { fileURLToPath } from 'url';
import { basename, dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let backendRoot = join(__dirname, '..', '..');
if (basename(backendRoot) === 'dist') {
  backendRoot = join(backendRoot, '..');
}
const appRoot = join(backendRoot, '..');

export function getAiServicePath(...parts: string[]) {
  return join(appRoot, 'ai_service', ...parts);
}

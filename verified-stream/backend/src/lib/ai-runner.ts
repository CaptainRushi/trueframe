import { spawn } from 'child_process';

/**
 * Run the AI detection process.
 * If AI_SERVICE_URL is set, it calls the remote REST endpoint.
 * Otherwise, it spawns a local Python process.
 */
export async function runAIScript(
  scriptPath: string,
  args: string[],
  timeoutMs: number = 120000
): Promise<any> {
  // Check if we should use a remote service (e.g., hosted on Lightning AI)
  if (process.env.AI_SERVICE_URL) {
    const filePath = args[0]; // Assuming the first argument is always the file path
    return callRemoteAIService(process.env.AI_SERVICE_URL, filePath, timeoutMs);
  }

  // Fallback to local spawn
  const pythonCmd = await getPythonCommand();
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath, ...args]);
    let stdout = '';
    let stderr = '';
    python.stdout.on('data', (d) => stdout += d.toString());
    python.stderr.on('data', (d) => stderr += d.toString());
    python.on('close', (code) => {
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          return resolve(parsed);
        }
      } catch (e) {
        // Parse failed
      }
      reject(new Error(`AI Error (exit ${code}): ${stderr || 'No valid JSON output'}`));
    });
    setTimeout(() => { python.kill(); reject(new Error('AI Timeout')); }, timeoutMs);
  });
}

/**
 * Call a remote AI service via REST API (multipart/form-data).
 */
async function callRemoteAIService(url: string, filePath: string, timeoutMs: number): Promise<any> {
  const { readFileSync } = await import('fs');
  const { basename } = await import('path');

  try {
    const fileBuffer = readFileSync(filePath);
    const fileName = basename(filePath);

    // Create form data
    const formData = new FormData();
    const blob = new Blob([fileBuffer]);
    formData.append('file', blob, fileName);

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(`${url.replace(/\/$/, '')}/verify`, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    clearTimeout(id);

    if (!response.ok) {
      throw new Error(`Remote AI Service Error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error('Remote AI Service Timeout');
    }
    throw error;
  }
}

/**
 * Detect available Python command.
 */
export async function getPythonCommand(): Promise<string> {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;
  const { execSync } = await import('child_process');
  try { execSync('python3 --version', { stdio: 'ignore' }); return 'python3'; }
  catch (e) {
    try { execSync('python --version', { stdio: 'ignore' }); return 'python'; }
    catch (e2) { throw new Error('Python not found'); }
  }
}

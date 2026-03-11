import { spawn } from 'child_process';

/**
 * Spawn a Python AI script and return parsed JSON output.
 * Fail-closed: rejects on non-zero exit, invalid JSON, or timeout.
 */
export async function runAIScript(
  scriptPath: string,
  args: string[],
  timeoutMs: number = 120000
): Promise<any> {
  const pythonCmd = await getPythonCommand();
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath, ...args]);
    let stdout = '';
    let stderr = '';
    python.stdout.on('data', (d) => stdout += d.toString());
    python.stderr.on('data', (d) => stderr += d.toString());
    python.on('close', (code) => {
      if (code !== 0) return reject(new Error(`AI Error ${code}: ${stderr}`));
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) resolve(JSON.parse(jsonMatch[0]));
        else reject(new Error('Invalid AI output'));
      } catch (e) { reject(e); }
    });
    setTimeout(() => { python.kill(); reject(new Error('AI Timeout')); }, timeoutMs);
  });
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

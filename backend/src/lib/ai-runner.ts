import { spawn } from 'child_process';

/**
 * Spawn a Python AI script and return parsed JSON output.
 * Fail-closed: rejects on timeout or if no valid JSON output is produced.
 * Tries to parse JSON from stdout even on non-zero exit codes,
 * since the AI scripts output valid fallback verdicts before exiting.
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
      // Always try to parse JSON from stdout first, regardless of exit code.
      // The AI scripts output valid fallback verdicts (REJECTED) even on errors.
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          return resolve(parsed);
        }
      } catch (e) {
        // JSON parse failed, fall through to error
      }
      // Only reject if we couldn't extract valid JSON
      reject(new Error(`AI Error (exit ${code}): ${stderr || 'No valid JSON output'}`));
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

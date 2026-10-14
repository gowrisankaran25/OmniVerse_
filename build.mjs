import { cpSync, mkdirSync, rmSync } from 'node:fs';

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist', { recursive: true });

cpSync('index.html', 'dist/index.html');
cpSync('styles.css', 'dist/styles.css');
cpSync('src', 'dist/src', { recursive: true });
cpSync('models', 'dist/models', { recursive: true });
try { cpSync('docs', 'dist/docs', { recursive: true }); } catch {}

try {
  mkdirSync('dist/node_modules/@mediapipe', { recursive: true });
  cpSync('node_modules/@mediapipe/tasks-vision', 'dist/node_modules/@mediapipe/tasks-vision', { recursive: true });
} catch {
  console.log('node_modules copy skipped');
}

console.log('Omniverse build finished: dist/ created');

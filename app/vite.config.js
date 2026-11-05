import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  // Read only this setting. Private backend variables never enter the browser bundle.
  const root = fileURLToPath(new URL('..', import.meta.url));
  const settings = loadEnv(mode, root, 'REPWISE_API_URL');
  const target = process.env.REPWISE_API_URL || settings.REPWISE_API_URL || 'http://127.0.0.1:8000';
  const proxy = { '/api': { target, changeOrigin: true } };
  return { server: { proxy }, preview: { proxy } };
});

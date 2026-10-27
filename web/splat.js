// Photoreal mode: Gaussian splat rendered with Spark (vendored, no CDN).
// Contract (CONTRACT.md): mountSplat({ THREE, scene, root, camera, renderer, meta }) -> { setVisible(bool), dispose() }
// The splat file is already baked into the scene.glb frame (metres, +Y up), so it is added under `root`
// without any transform and calibration scaling (root.scale) applies to it for free.
import { SparkRenderer, SplatMesh } from './vendor/spark/spark.module.min.js';

export async function mountSplat({ THREE, scene, root, camera, renderer, meta }) {
  if (!meta?.splat?.url) throw new Error('meta.splat.url missing');

  // Spark sorts and draws splats from its own renderer object; it must live in the scene.
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const splat = new SplatMesh({ url: meta.splat.url });
  // Measurement picking must only ever hit the metric mesh, never the splat.
  splat.raycast = () => {};
  splat.visible = false;
  root.add(splat);

  try {
    await splat.initialized;
  } catch (err) {
    scene.remove(spark);
    root.remove(splat);
    throw err;
  }

  return {
    setVisible(visible) {
      splat.visible = !!visible;
    },
    dispose() {
      root.remove(splat);
      scene.remove(spark);
      splat.dispose();
    },
  };
}

// import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
// import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js";
// import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/OBJLoader.js";
// import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

// let camera, scene, renderer, mesh, pointCloud, controls;

// init();
// animate();

// function init() {
//   scene = new THREE.Scene();
//   camera = new THREE.PerspectiveCamera(
//     75,
//     window.innerWidth / window.innerHeight,
//     0.1,
//     1000
//   );
//   renderer = new THREE.WebGLRenderer();
//   renderer.setSize(window.innerWidth, window.innerHeight);
//   document.body.appendChild(renderer.domElement);

//   // Lighting
//   const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // soft white light
//   scene.add(ambientLight);
//   const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
//   directionalLight.position.set(0, 1, 0);
//   scene.add(directionalLight);

//   // Controls
//   controls = new OrbitControls(camera, renderer.domElement);
//   controls.enableDamping = true;
//   controls.dampingFactor = 0.25;
//   controls.screenSpacePanning = false;
//   controls.minDistance = 0.1;
//   controls.maxDistance = 100;

//   // OBJ Mesh
//   const objLoader = new OBJLoader();
//   objLoader.load("result.obj", function (obj) {
//     mesh = obj;
//     mesh.traverse((child) => {
//       if (child.isMesh) {
//         // Set material to use vertex colors if available
//         child.material.vertexColors = THREE.VertexColors;
//         // Set original color
//         child.material.color.set(0xffffff);
//       }
//     });
//     scene.add(mesh);
//   });

//   // PLY Point Cloud
//   const plyLoader = new PLYLoader();
//   plyLoader.load("result.ply", function (geometry) {
//     geometry.computeVertexNormals();
//     const material = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 });
//     // Set original color
//     material.color.set(0xffffff);
//     pointCloud = new THREE.Points(geometry, material);
//     scene.add(pointCloud);
//   });

//   camera.position.z = 5;
// }

// function animate() {
//   requestAnimationFrame(animate);
//   controls.update(); // Update controls every frame
//   renderer.render(scene, camera);
// }

// import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
// import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js";
// import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/OBJLoader.js";
// import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

// let camera, scene, renderer, mesh, pointCloud, controls;

// init();
// animate();

// function init() {
//   scene = new THREE.Scene();

//   // Camera setup
//   camera = new THREE.PerspectiveCamera(
//     75,
//     window.innerWidth / window.innerHeight,
//     0.1,
//     1000
//   );
//   camera.position.set(0, 0, 10);

//   // Renderer setup
//   renderer = new THREE.WebGLRenderer();
//   renderer.setSize(window.innerWidth, window.innerHeight);
//   document.body.appendChild(renderer.domElement);

//   // Orbit controls setup
//   controls = new OrbitControls(camera, renderer.domElement);
//   controls.enableDamping = true;
//   controls.dampingFactor = 0.25;
//   controls.enableZoom = true;

//   // Lighting
//   const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
//   scene.add(ambientLight);
//   const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//   directionalLight.position.set(0, 1, 0);
//   scene.add(directionalLight);

//   // Axes helper for orientation
//   const axesHelper = new THREE.AxesHelper(5);
//   scene.add(axesHelper);

//   // OBJ Mesh
//   const objLoader = new OBJLoader();
//   objLoader.load("result.obj", function (obj) {
//     mesh = obj;
//     scene.add(mesh);
//   });

//   // PLY Point Cloud
//   const plyLoader = new PLYLoader();
//   plyLoader.load("result.ply", function (geometry) {
//     const material = new THREE.PointsMaterial({
//       vertexColors: THREE.VertexColors,
//       size: 0.05,
//     });
//     pointCloud = new THREE.Points(geometry, material);
//     scene.add(pointCloud);
//   });

//   // Resize handling
//   window.addEventListener("resize", onWindowResize);
// }

// function animate() {
//   requestAnimationFrame(animate);
//   controls.update(); // Update controls
//   renderer.render(scene, camera);
// }

// function onWindowResize() {
//   camera.aspect = window.innerWidth / window.innerHeight;
//   camera.updateProjectionMatrix();
//   renderer.setSize(window.innerWidth, window.innerHeight);
// }

// --------------------------------------------------------------------------------------
// for showing each points separately

// import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
// import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js";
// import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/OBJLoader.js";
// import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

// let camera, scene, renderer, controls;

// init();
// animate();

// function init() {
//   scene = new THREE.Scene();

//   // Camera setup
//   camera = new THREE.PerspectiveCamera(
//     75,
//     window.innerWidth / window.innerHeight,
//     0.1,
//     1000
//   );
//   camera.position.set(0, 0, 10);

//   // Renderer setup
//   renderer = new THREE.WebGLRenderer();
//   renderer.setSize(window.innerWidth, window.innerHeight);
//   document.body.appendChild(renderer.domElement);

//   // Orbit controls setup
//   controls = new OrbitControls(camera, renderer.domElement);
//   controls.enableDamping = true;
//   controls.dampingFactor = 0.25;
//   controls.enableZoom = true;

//   // Lighting
//   const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
//   scene.add(ambientLight);
//   const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//   directionalLight.position.set(0, 1, 0);
//   scene.add(directionalLight);

//   // Axes helper for orientation
//   const axesHelper = new THREE.AxesHelper(5);
//   scene.add(axesHelper);

//   // Load OBJ
//   const objLoader = new OBJLoader();
//   objLoader.load("result.obj", function (obj) {
//     obj.traverse(function (child) {
//       if (child instanceof THREE.Mesh) {
//         const pointsMaterial = new THREE.PointsMaterial({ size: 0.05 });
//         const points = new THREE.Points(child.geometry, pointsMaterial);
//         scene.add(points);
//       }
//     });
//   });

//   // Load PLY
//   const plyLoader = new PLYLoader();
//   plyLoader.load("result.ply", function (geometry) {
//     const pointsMaterial = new THREE.PointsMaterial({ size: 0.05 });
//     const points = new THREE.Points(geometry, pointsMaterial);
//     scene.add(points);
//   });

//   // Resize handling
//   window.addEventListener("resize", onWindowResize);
// }

// function animate() {
//   requestAnimationFrame(animate);
//   controls.update(); // Update controls
//   renderer.render(scene, camera);
// }

// function onWindowResize() {
//   camera.aspect = window.innerWidth / window.innerHeight;
//   camera.updateProjectionMatrix();
//   renderer.setSize(window.innerWidth, window.innerHeight);
// }

//---------------------------------------------------UNCOMMENT THIS-----------------------

// import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
// import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js";
// import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/OBJLoader.js";
// import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

// let camera, scene, renderer, controls;

// init();
// animate();

// function init() {
//   scene = new THREE.Scene();

//   // Camera setup
//   camera = new THREE.PerspectiveCamera(
//     75,
//     window.innerWidth / window.innerHeight,
//     0.1,
//     1000
//   );
//   camera.position.set(0, 0, 10);

//   // Renderer setup
//   renderer = new THREE.WebGLRenderer();
//   renderer.setSize(window.innerWidth, window.innerHeight);
//   document.body.appendChild(renderer.domElement);

//   // Orbit controls setup
//   controls = new OrbitControls(camera, renderer.domElement);
//   controls.enableDamping = true;
//   controls.dampingFactor = 0.25;
//   controls.enableZoom = true;

//   // Lighting
//   const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
//   scene.add(ambientLight);
//   const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
//   directionalLight.position.set(0, 1, 0);
//   scene.add(directionalLight);

//   // Axes helper for orientation
//   const axesHelper = new THREE.AxesHelper(5);
//   scene.add(axesHelper);

// Load OBJ
// const objLoader = new OBJLoader();
// objLoader.load("result.obj", function (obj) {
//   obj.traverse(function (child) {
//     if (child instanceof THREE.Mesh) {
//       // Ensure vertex colors are available
//       if (!child.geometry.attributes.color) {
//         console.warn("OBJ file does not contain vertex colors.");
//         return;
//       }
//       const pointsMaterial = new THREE.PointsMaterial({
//         size: 0.05,
//         vertexColors: THREE.VertexColors,
//       });
//       const points = new THREE.Points(child.geometry, pointsMaterial);
//       scene.add(points);
//     }
//   });
// });

//---------------------------------------------------COMMENT THIS-----------------------

//   const plyLoader2 = new PLYLoader();
//   plyLoader2.load("result1.ply", function (geometry) {
//     // Ensure vertex colors are available
//     if (!geometry.attributes.color) {
//       console.warn("PLY file does not contain vertex colors.");
//       return;
//     }
//     const pointsMaterial = new THREE.PointsMaterial({
//       size: 0.05,
//       vertexColors: THREE.VertexColors,
//     });
//     const points = new THREE.Points(geometry, pointsMaterial);
//     scene.add(points);
//   });

//---------------------------------------------------UNCOMMENT THIS-----------------------

//   // Load PLY
//   const plyLoader = new PLYLoader();
//   plyLoader.load("result2.ply", function (geometry) {
//     // Ensure vertex colors are available
//     if (!geometry.attributes.color) {
//       console.warn("PLY file does not contain vertex colors.");
//       return;
//     }
//     const pointsMaterial = new THREE.PointsMaterial({
//       size: 0.05,
//       vertexColors: THREE.VertexColors,
//     });
//     const points = new THREE.Points(geometry, pointsMaterial);
//     scene.add(points);
//   });

//   // Resize handling
//   window.addEventListener("resize", onWindowResize);
// }

// function animate() {
//   requestAnimationFrame(animate);
//   controls.update(); // Update controls
//   renderer.render(scene, camera);
// }

// function onWindowResize() {
//   camera.aspect = window.innerWidth / window.innerHeight;
//   camera.updateProjectionMatrix();
//   renderer.setSize(window.innerWidth, window.innerHeight);
// }

//---------------------------------------------------COMMENT THIS-----------------------

// import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
// import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/GLTFLoader.js";
// import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

// // Set up renderer
// const renderer = new THREE.WebGLRenderer();
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// // Set up camera
// const camera = new THREE.PerspectiveCamera(
//   75,
//   window.innerWidth / window.innerHeight,
//   0.1,
//   1000
// );
// camera.position.set(0, 0, 10);

// // Set up scene
// const scene = new THREE.Scene();

// // Load PLY
// const plyLoader = new PLYLoader();
// plyLoader.load("result1.ply", function (geometry) {
//   // Ensure vertex colors are available
//   if (!geometry.attributes.color) {
//     console.warn("PLY file does not contain vertex colors.");
//     return;
//   }
//   const pointsMaterial = new THREE.PointsMaterial({
//     size: 0.05,
//     vertexColors: THREE.VertexColors,
//   });
//   const points = new THREE.Points(geometry, pointsMaterial);
//   scene.add(points);
// });

// const plyLoader2 = new PLYLoader();
// plyLoader2.load("result2.ply", function (geometry) {
//   // Ensure vertex colors are available
//   if (!geometry.attributes.color) {
//     console.warn("PLY file does not contain vertex colors.");
//     return;
//   }
//   const pointsMaterial = new THREE.PointsMaterial({
//     size: 0.05,
//     vertexColors: THREE.VertexColors,
//   });
//   const points = new THREE.Points(geometry, pointsMaterial);
//   scene.add(points);
// });

// // // Load mesh
// // const meshLoader = new GLTFLoader();
// // meshLoader.load("result2.ply", function (gltf) {
// //   const mesh = gltf.scene;
// //   scene.add(mesh);
// // });

// // Resize handling
//   window.addEventListener("resize", onWindowResize);
// }

// function animate() {
//   requestAnimationFrame(animate);
//   controls.update(); // Update controls
//   renderer.render(scene, camera);
// }

// function onWindowResize() {
//   camera.aspect = window.innerWidth / window.innerHeight;
//   camera.updateProjectionMatrix();
//   renderer.setSize(window.innerWidth, window.innerHeight);
// }

// // Enable depth testing
// renderer.setClearColor(0x000000);
// renderer.setClearAlpha(0);
// renderer.autoClear = false;
// renderer.sortObjects = false;

// // Set up lighting
// const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
// scene.add(ambientLight);

// const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
// directionalLight.position.set(0, 1, 0);
// scene.add(directionalLight);

// // Render function
// function render() {
//   renderer.clear();
//   renderer.render(scene, camera);
//   requestAnimationFrame(render);
// }
// render();

//-------------------------------------------------------------------------------------

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.127.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.127.0/examples/jsm/loaders/PLYLoader.js";

let camera, scene, renderer, controls;

init();
animate();

function init() {
  scene = new THREE.Scene();

  // Camera setup
  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 0, 10);

  // Renderer setup
  renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById("viewer-container").appendChild(renderer.domElement);

  // Orbit controls setup
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.enableZoom = true;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
  directionalLight.position.set(0, 1, 0);
  scene.add(directionalLight);

  // Axes helper for orientation
  const axesHelper = new THREE.AxesHelper(5);
  scene.add(axesHelper);

  setTimeout(function () {
    const plyLoader = new PLYLoader();
    plyLoader.load("result2.ply", function (geometry) {
      // Ensure vertex colors are available
      if (!geometry.attributes.color) {
        console.warn("PLY file does not contain vertex colors.");
        return;
      }
      const pointsMaterial = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: THREE.VertexColors,
      });
      const points = new THREE.Points(geometry, pointsMaterial);
      scene.add(points);
    });
  }, 15000);
  // const plyLoader2 = new PLYLoader();
  // plyLoader2.load("result.ply", function (geometry) {
  //   // Ensure vertex colors are available
  //   if (!geometry.attributes.color) {
  //     console.warn("PLY file does not contain vertex colors.");
  //     return;
  //   }
  //   const pointsMaterial = new THREE.PointsMaterial({
  //     size: 0.05,
  //     vertexColors: THREE.VertexColors,
  //   });
  //   const points = new THREE.Points(geometry, pointsMaterial);
  //   scene.add(points);
  // });

  // // Load PLY
  // const plyLoader = new PLYLoader();
  // plyLoader.load("result2.ply", function (geometry) {
  //   // Ensure vertex colors are available
  //   if (!geometry.attributes.color) {
  //     console.warn("PLY file does not contain vertex colors.");
  //     return;
  //   }
  //   const pointsMaterial = new THREE.PointsMaterial({
  //     size: 0.05,
  //     vertexColors: THREE.VertexColors,
  //   });
  //   const points = new THREE.Points(geometry, pointsMaterial);
  //   scene.add(points);
  // });

  // Resize handling
  window.addEventListener("resize", onWindowResize);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update(); // Update controls
  renderer.render(scene, camera);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

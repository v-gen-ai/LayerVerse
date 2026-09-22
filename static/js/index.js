'use strict';
(() => {
  const groups = [
    {
      "id": "flux-fireflow",
      "label": "FLUX.1-[dev] · FireFlow",
      "backbone": "FLUX.1-[dev]",
      "solver": "FireFlow",
      "methods": [
        "Source image",
        "RF-inversion",
        "FireFlow",
        "FlowEdit",
        "UniEdit-Flow",
        "DNAEdit",
        "FireFlow + LayerVerse"
      ],
      "baseline": 2,
      "examples": [
        {
          "prompt": "Car → motorcycle",
          "task": "Change object",
          "images": [
            "static/images/flux-fireflow-01-0.jpg",
            "static/images/flux-fireflow-01-1.jpg",
            "static/images/flux-fireflow-01-2.jpg",
            "static/images/flux-fireflow-01-3.jpg",
            "static/images/flux-fireflow-01-4.jpg",
            "static/images/flux-fireflow-01-5.jpg",
            "static/images/flux-fireflow-01-6.jpg"
          ]
        },
        {
          "prompt": "Snow → leaves",
          "task": "Background",
          "images": [
            "static/images/flux-fireflow-02-0.jpg",
            "static/images/flux-fireflow-02-1.jpg",
            "static/images/flux-fireflow-02-2.jpg",
            "static/images/flux-fireflow-02-3.jpg",
            "static/images/flux-fireflow-02-4.jpg",
            "static/images/flux-fireflow-02-5.jpg",
            "static/images/flux-fireflow-02-6.jpg"
          ]
        },
        {
          "prompt": "Add hair band",
          "task": "Add",
          "images": [
            "static/images/flux-fireflow-03-0.jpg",
            "static/images/flux-fireflow-03-1.jpg",
            "static/images/flux-fireflow-03-2.jpg",
            "static/images/flux-fireflow-03-3.jpg",
            "static/images/flux-fireflow-03-4.jpg",
            "static/images/flux-fireflow-03-5.jpg",
            "static/images/flux-fireflow-03-6.jpg"
          ]
        },
        {
          "prompt": "Walk → run",
          "task": "Pose",
          "images": [
            "static/images/flux-fireflow-04-0.jpg",
            "static/images/flux-fireflow-04-1.jpg",
            "static/images/flux-fireflow-04-2.jpg",
            "static/images/flux-fireflow-04-3.jpg",
            "static/images/flux-fireflow-04-4.jpg",
            "static/images/flux-fireflow-04-5.jpg",
            "static/images/flux-fireflow-04-6.jpg"
          ]
        },
        {
          "prompt": "Change material to fabric",
          "task": "Change material",
          "images": [
            "static/images/flux-fireflow-05-0.jpg",
            "static/images/flux-fireflow-05-1.jpg",
            "static/images/flux-fireflow-05-2.jpg",
            "static/images/flux-fireflow-05-3.jpg",
            "static/images/flux-fireflow-05-4.jpg",
            "static/images/flux-fireflow-05-5.jpg",
            "static/images/flux-fireflow-05-6.jpg"
          ]
        },
        {
          "prompt": "Pink → yellow",
          "task": "Change color",
          "images": [
            "static/images/flux-fireflow-06-0.jpg",
            "static/images/flux-fireflow-06-1.jpg",
            "static/images/flux-fireflow-06-2.jpg",
            "static/images/flux-fireflow-06-3.jpg",
            "static/images/flux-fireflow-06-4.jpg",
            "static/images/flux-fireflow-06-5.jpg",
            "static/images/flux-fireflow-06-6.jpg"
          ]
        },
        {
          "prompt": "Delete rainbow",
          "task": "Delete",
          "images": [
            "static/images/flux-fireflow-07-0.jpg",
            "static/images/flux-fireflow-07-1.jpg",
            "static/images/flux-fireflow-07-2.jpg",
            "static/images/flux-fireflow-07-3.jpg",
            "static/images/flux-fireflow-07-4.jpg",
            "static/images/flux-fireflow-07-5.jpg",
            "static/images/flux-fireflow-07-6.jpg"
          ]
        }
      ]
    },
    {
      "id": "flux-rfsolver",
      "label": "FLUX.1-[dev] · RF-Solver",
      "backbone": "FLUX.1-[dev]",
      "solver": "RF-Solver",
      "methods": [
        "Source image",
        "RF-inversion",
        "RF-Solver",
        "FlowEdit",
        "UniEdit-Flow",
        "DNAEdit",
        "RF-Solver + LayerVerse"
      ],
      "baseline": 2,
      "examples": [
        {
          "prompt": "Fruits → pizza",
          "task": "Change object",
          "images": [
            "static/images/flux-rfsolver-01-0.jpg",
            "static/images/flux-rfsolver-01-1.jpg",
            "static/images/flux-rfsolver-01-2.jpg",
            "static/images/flux-rfsolver-01-3.jpg",
            "static/images/flux-rfsolver-01-4.jpg",
            "static/images/flux-rfsolver-01-5.jpg",
            "static/images/flux-rfsolver-01-6.jpg"
          ]
        },
        {
          "prompt": "Rocks → boat",
          "task": "Change object",
          "images": [
            "static/images/flux-rfsolver-02-0.jpg",
            "static/images/flux-rfsolver-02-1.jpg",
            "static/images/flux-rfsolver-02-2.jpg",
            "static/images/flux-rfsolver-02-3.jpg",
            "static/images/flux-rfsolver-02-4.jpg",
            "static/images/flux-rfsolver-02-5.jpg",
            "static/images/flux-rfsolver-02-6.jpg"
          ]
        },
        {
          "prompt": "Add balloons",
          "task": "Add",
          "images": [
            "static/images/flux-rfsolver-03-0.jpg",
            "static/images/flux-rfsolver-03-1.jpg",
            "static/images/flux-rfsolver-03-2.jpg",
            "static/images/flux-rfsolver-03-3.jpg",
            "static/images/flux-rfsolver-03-4.jpg",
            "static/images/flux-rfsolver-03-5.jpg",
            "static/images/flux-rfsolver-03-6.jpg"
          ]
        },
        {
          "prompt": "Calm → laughing",
          "task": "Expression",
          "images": [
            "static/images/flux-rfsolver-04-0.jpg",
            "static/images/flux-rfsolver-04-1.jpg",
            "static/images/flux-rfsolver-04-2.jpg",
            "static/images/flux-rfsolver-04-3.jpg",
            "static/images/flux-rfsolver-04-4.jpg",
            "static/images/flux-rfsolver-04-5.jpg",
            "static/images/flux-rfsolver-04-6.jpg"
          ]
        },
        {
          "prompt": "Add sky with stars",
          "task": "Background",
          "images": [
            "static/images/flux-rfsolver-05-0.jpg",
            "static/images/flux-rfsolver-05-1.jpg",
            "static/images/flux-rfsolver-05-2.jpg",
            "static/images/flux-rfsolver-05-3.jpg",
            "static/images/flux-rfsolver-05-4.jpg",
            "static/images/flux-rfsolver-05-5.jpg",
            "static/images/flux-rfsolver-05-6.jpg"
          ]
        },
        {
          "prompt": "Goat → horse",
          "task": "Change object",
          "images": [
            "static/images/flux-rfsolver-06-0.jpg",
            "static/images/flux-rfsolver-06-1.jpg",
            "static/images/flux-rfsolver-06-2.jpg",
            "static/images/flux-rfsolver-06-3.jpg",
            "static/images/flux-rfsolver-06-4.jpg",
            "static/images/flux-rfsolver-06-5.jpg",
            "static/images/flux-rfsolver-06-6.jpg"
          ]
        },
        {
          "prompt": "Remove kiwi",
          "task": "Delete",
          "images": [
            "static/images/flux-rfsolver-07-0.jpg",
            "static/images/flux-rfsolver-07-1.jpg",
            "static/images/flux-rfsolver-07-2.jpg",
            "static/images/flux-rfsolver-07-3.jpg",
            "static/images/flux-rfsolver-07-4.jpg",
            "static/images/flux-rfsolver-07-5.jpg",
            "static/images/flux-rfsolver-07-6.jpg"
          ]
        }
      ]
    },
    {
      "id": "sd35-fireflow",
      "label": "SD3.5-medium · FireFlow",
      "backbone": "SD3.5-medium",
      "solver": "FireFlow",
      "methods": [
        "Source image",
        "FTEdit",
        "FlowEdit",
        "FSI-Edit",
        "DNAEdit",
        "FireFlow + LayerVerse"
      ],
      "baseline": 4,
      "examples": [
        {
          "prompt": "Cat → dog",
          "task": "Change object",
          "images": [
            "static/images/sd35-fireflow-01-0.jpg",
            "static/images/sd35-fireflow-01-1.jpg",
            "static/images/sd35-fireflow-01-2.jpg",
            "static/images/sd35-fireflow-01-3.jpg",
            "static/images/sd35-fireflow-01-4.jpg",
            "static/images/sd35-fireflow-01-5.jpg"
          ]
        },
        {
          "prompt": "Change shape to X",
          "task": "Shape",
          "images": [
            "static/images/sd35-fireflow-02-0.jpg",
            "static/images/sd35-fireflow-02-1.jpg",
            "static/images/sd35-fireflow-02-2.jpg",
            "static/images/sd35-fireflow-02-3.jpg",
            "static/images/sd35-fireflow-02-4.jpg",
            "static/images/sd35-fireflow-02-5.jpg"
          ]
        },
        {
          "prompt": "Young → old",
          "task": "Attribute",
          "images": [
            "static/images/sd35-fireflow-03-0.jpg",
            "static/images/sd35-fireflow-03-1.jpg",
            "static/images/sd35-fireflow-03-2.jpg",
            "static/images/sd35-fireflow-03-3.jpg",
            "static/images/sd35-fireflow-03-4.jpg",
            "static/images/sd35-fireflow-03-5.jpg"
          ]
        },
        {
          "prompt": "Change material to jewels",
          "task": "Change material",
          "images": [
            "static/images/sd35-fireflow-04-0.jpg",
            "static/images/sd35-fireflow-04-1.jpg",
            "static/images/sd35-fireflow-04-2.jpg",
            "static/images/sd35-fireflow-04-3.jpg",
            "static/images/sd35-fireflow-04-4.jpg",
            "static/images/sd35-fireflow-04-5.jpg"
          ]
        },
        {
          "prompt": "Add car",
          "task": "Add",
          "images": [
            "static/images/sd35-fireflow-05-0.jpg",
            "static/images/sd35-fireflow-05-1.jpg",
            "static/images/sd35-fireflow-05-2.jpg",
            "static/images/sd35-fireflow-05-3.jpg",
            "static/images/sd35-fireflow-05-4.jpg",
            "static/images/sd35-fireflow-05-5.jpg"
          ]
        },
        {
          "prompt": "Add cat",
          "task": "Add",
          "images": [
            "static/images/sd35-fireflow-06-0.jpg",
            "static/images/sd35-fireflow-06-1.jpg",
            "static/images/sd35-fireflow-06-2.jpg",
            "static/images/sd35-fireflow-06-3.jpg",
            "static/images/sd35-fireflow-06-4.jpg",
            "static/images/sd35-fireflow-06-5.jpg"
          ]
        },
        {
          "prompt": "Mountains → sea",
          "task": "Background",
          "images": [
            "static/images/sd35-fireflow-07-0.jpg",
            "static/images/sd35-fireflow-07-1.jpg",
            "static/images/sd35-fireflow-07-2.jpg",
            "static/images/sd35-fireflow-07-3.jpg",
            "static/images/sd35-fireflow-07-4.jpg",
            "static/images/sd35-fireflow-07-5.jpg"
          ]
        }
      ]
    }
  ];
  const ablations = [
    {
      "prompt": "Rabbit → cat",
      "images": [
        "static/images/roles-1-0.jpg",
        "static/images/roles-1-1.jpg",
        "static/images/roles-1-2.jpg",
        "static/images/roles-1-3.jpg",
        "static/images/roles-1-4.jpg"
      ],
      "methods": [
        "Source image",
        "FireFlow",
        "Only global KV",
        "Only masked KV",
        "FireFlow + LayerVerse"
      ]
    },
    {
      "prompt": "Mug → glass",
      "images": [
        "static/images/roles-2-0.jpg",
        "static/images/roles-2-1.jpg",
        "static/images/roles-2-2.jpg",
        "static/images/roles-2-3.jpg",
        "static/images/roles-2-4.jpg"
      ],
      "methods": [
        "Source image",
        "FireFlow",
        "Only global KV",
        "Only masked KV",
        "FireFlow + LayerVerse"
      ]
    },
    {
      "prompt": "Standing → sitting",
      "images": [
        "static/images/roles-3-0.jpg",
        "static/images/roles-3-1.jpg",
        "static/images/roles-3-2.jpg",
        "static/images/roles-3-3.jpg",
        "static/images/roles-3-4.jpg"
      ],
      "methods": [
        "Source image",
        "FireFlow",
        "Only global KV",
        "Only masked KV",
        "FireFlow + LayerVerse"
      ]
    }
  ];
  const captions = {
    "RF-inversion": "static/images/captions/rf-inversion.svg",
    "FireFlow": "static/images/captions/fireflow.svg",
    "FlowEdit": "static/images/captions/flowedit.svg",
    "UniEdit-Flow": "static/images/captions/uniedit-flow.svg",
    "DNAEdit": "static/images/captions/dnaedit.svg",
    "FireFlow + LayerVerse": "static/images/captions/fireflow-layerverse.svg",
    "Car → motorcycle": "static/images/captions/car-to-motorcycle.svg",
    "Snow → leaves": "static/images/captions/snow-to-leaves.svg",
    "Add hair band": "static/images/captions/add-hair-band.svg",
    "Walk → run": "static/images/captions/walk-to-run.svg",
    "Change material to fabric": "static/images/captions/change-material-to-fabric.svg",
    "Pink → yellow": "static/images/captions/pink-to-yellow.svg",
    "Delete rainbow": "static/images/captions/delete-rainbow.svg",
    "RF-Solver": "static/images/captions/rf-solver.svg",
    "RF-Solver + LayerVerse": "static/images/captions/rf-solver-layerverse.svg",
    "Fruits → pizza": "static/images/captions/fruits-to-pizza.svg",
    "Rocks → boat": "static/images/captions/rocks-to-boat.svg",
    "Add balloons": "static/images/captions/add-balloons.svg",
    "Calm → laughing": "static/images/captions/calm-to-laughing.svg",
    "Add sky with stars": "static/images/captions/add-sky-with-stars.svg",
    "Goat → horse": "static/images/captions/goat-to-horse.svg",
    "Remove kiwi": "static/images/captions/remove-kiwi.svg",
    "FTEdit": "static/images/captions/ftedit.svg",
    "FSI-Edit": "static/images/captions/fsi-edit.svg",
    "Cat → dog": "static/images/captions/cat-to-dog.svg",
    "Change shape to X": "static/images/captions/change-shape-to-x.svg",
    "Young → old": "static/images/captions/young-to-old.svg",
    "Change material to jewels": "static/images/captions/change-material-to-jewels.svg",
    "Add car": "static/images/captions/add-car.svg",
    "Add cat": "static/images/captions/add-cat.svg",
    "Mountains → sea": "static/images/captions/mountains-to-sea.svg",
    "Only global KV": "static/images/captions/only-global-kv.svg",
    "Only masked KV": "static/images/captions/only-masked-kv.svg",
    "FireFlow +\nLayerVerse": "static/images/captions/fireflow-layerverse-stacked.svg"
  };
  const $ = id => document.getElementById(id);
  let groupIndex = 0, exampleIndex = 0, baselineIndex = groups[0]?.baseline || 2, view = 'columns';
  function setCaption(element, text) {
    const source = text.toLowerCase() === 'source image';
    element.classList.toggle('source-label', source);
    if (source) {
      element.textContent = 'Source image';
      return;
    }
    const path = captions[text];
    if (!path) {
      element.textContent = text;
      return;
    }
    const copy = document.createElement('span');
    copy.className = 'caption-copy';
    copy.textContent = text;
    const image = document.createElement('img');
    image.className = 'caption-art';
    image.src = path;
    image.alt = '';
    image.setAttribute('aria-hidden', 'true');
    image.addEventListener('error', () => { element.textContent = text; }, { once: true });
    element.replaceChildren(copy, image);
  }
  function imageButton(path, caption) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'image-button';
    button.dataset.zoom = path;
    button.dataset.caption = caption;
    button.setAttribute('aria-label', `Enlarge: ${caption}`);
    const image = document.createElement('img');
    image.src = path;
    image.alt = caption;
    image.width = 512;
    image.height = 512;
    image.decoding = 'async';
    button.append(image);
    return button;
  }
  function card(path, label, prompt, ours = false) {
    const el = document.createElement('div');
    el.className = `comparison-card${ours ? ' ours' : ''}`;
    const title = document.createElement('p');
    title.className = 'image-label';
    setCaption(title, label);
    el.append(title, imageButton(path, `${prompt} — ${label}`));
    return el;
  }
  function swipe(source, target, label, prompt, ours = false) {
    const figure = document.createElement('figure');
    figure.className = `comparison-card${ours ? ' ours' : ''}`;
    const block = document.createElement('div');
    block.className = 'compare-swipe';
    block.style.setProperty('--split', '50%');
    const result = document.createElement('img');
    result.src = target;
    result.alt = `${prompt} — ${label}`;
    const original = document.createElement('img');
    original.className = 'source-over';
    original.src = source;
    original.alt = 'Source image';
    const line = document.createElement('span');
    line.className = 'swipe-divider';
    line.setAttribute('aria-hidden', 'true');
    const range = document.createElement('input');
    range.type = 'range';
    range.min = '0';
    range.max = '100';
    range.value = '50';
    range.className = 'swipe-range';
    range.setAttribute('aria-label', `Reveal ${label} versus source`);
    range.addEventListener('input', () => block.style.setProperty('--split', `${range.value}%`));
    const left = document.createElement('span');
    left.className = 'swipe-label left source-label';
    left.textContent = 'Source';
    const right = document.createElement('span');
    right.className = 'swipe-label right';
    if (label.includes('LayerVerse')) {
      right.append(label.replace('LayerVerse', ''));
      const name = document.createElement('strong');
      name.className = 'lv-name';
      name.textContent = 'LayerVerse';
      right.append(name);
    }
    else {
      right.textContent = label;
    }
    block.append(result, original, line, left, right, range);
    const caption = document.createElement('figcaption');
    caption.className = 'image-label';
    setCaption(caption, label);
    figure.append(caption, block);
    return figure;
  }
  function renderGallery(newGroup = false) {
    const g = groups[groupIndex], e = g.examples[exampleIndex];
    $('gallery-task').textContent = e.task;
    setCaption($('gallery-prompt'), e.prompt);
    $('gallery-count').textContent = `${exampleIndex + 1} / ${g.examples.length}`;
    $('gallery-status').textContent = `${g.backbone}, ${g.solver}. Example ${exampleIndex + 1} of ${g.examples.length}: ${e.prompt}.`;
    $('gallery-images').replaceChildren(card(e.images[0], 'Source image', e.prompt), card(e.images[baselineIndex], g.methods[baselineIndex], e.prompt), card(e.images.at(-1), `${g.solver} + LayerVerse`, e.prompt, true));
    const ref = card(e.images[0], 'Source image', e.prompt);
    ref.classList.add('reference-view');
    $('swipe-images').replaceChildren(ref, swipe(e.images[0], e.images[baselineIndex], g.methods[baselineIndex], e.prompt), swipe(e.images[0], e.images.at(-1), `${g.solver} + LayerVerse`, e.prompt, true));
    if (newGroup) {
      const select = $('baseline-select');
      select.replaceChildren();
      g.methods.slice(1, -1).forEach((label, index) => {
        const option = document.createElement('option');
        option.value = String(index + 1);
        option.textContent = label;
        select.append(option);
      });
      select.value = String(baselineIndex);
      $('gallery-thumbnails').replaceChildren();
      g.examples.forEach((item, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'thumbnail';
        button.setAttribute('aria-label', `Example ${index + 1}: ${item.prompt}`);
        button.addEventListener('click', () => { exampleIndex = index; renderGallery(); });
        const img = document.createElement('img');
        img.src = item.images.at(-1);
        img.alt = '';
        img.width = 44;
        img.height = 44;
        img.loading = 'lazy';
        button.append(img);
        $('gallery-thumbnails').append(button);
      });
    }
    [...$('gallery-thumbnails').children].forEach((b, i) => b.setAttribute('aria-pressed', String(i === exampleIndex)));
    document.querySelectorAll('[data-gallery-group]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.galleryGroup) === groupIndex)));
    const note = $('gallery-note');
    note.hidden = groupIndex !== 2;
    note.textContent = 'SD3.5-medium: comparison to other editors; unmodified FireFlow was not evaluated.';
    const all = $('all-methods-grid');
    all.replaceChildren();
    all.style.gridTemplateColumns = `repeat(${g.methods.length}, minmax(0, 1fr))`;
    g.methods.forEach((label, i) => all.append(card(e.images[i], label, e.prompt, i === g.methods.length - 1)));
    applyView();
    const next = g.examples[(exampleIndex + 1) % g.examples.length];
    [0, baselineIndex, next.images.length - 1].forEach(i => { const img = new Image(); img.src = next.images[i]; });
  }
  function applyView() {
    $('gallery-viewport').hidden = view !== 'columns';
    $('swipe-images').hidden = view !== 'swipe';
    $('gallery-scroll-hint').hidden = view !== 'columns';
    document.querySelectorAll('[data-gallery-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.galleryView === view)));
  }
  function move(direction) { exampleIndex = (exampleIndex + direction + groups[groupIndex].examples.length) % groups[groupIndex].examples.length; renderGallery(); }
  if (groups.length && $('gallery-images')) {
    document.querySelectorAll('[data-gallery-group]').forEach(b => b.addEventListener('click', () => {
      groupIndex = Number(b.dataset.galleryGroup);
      exampleIndex = 0;
      baselineIndex = groups[groupIndex].baseline;
      renderGallery(true);
    }));
    $('gallery-prev').addEventListener('click', () => move(-1));
    $('gallery-next').addEventListener('click', () => move(1));
    $('baseline-select').addEventListener('change', event => {
      const i = Number(event.target.value);
      if (i >= 1 && i < groups[groupIndex].methods.length - 1) {
        baselineIndex = i;
        renderGallery();
      }
    });
    $('gallery-shell').addEventListener('keydown', event => {
      if (event.target.closest('select,input,textarea'))
        return;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        move(event.key === 'ArrowRight' ? 1 : -1);
      }
    });
    document.querySelectorAll('[data-gallery-view]').forEach(b => b.addEventListener('click', () => { view = b.dataset.galleryView; applyView(); }));
    $('all-methods-toggle').addEventListener('click', () => {
      const panel = $('all-methods-panel');
      panel.hidden = !panel.hidden;
      $('all-methods-toggle').setAttribute('aria-expanded', String(!panel.hidden));
      $('all-methods-toggle').textContent = panel.hidden ? 'Show all methods for this example ↓' : 'Hide full comparison ↑';
    });
    renderGallery(true);
  }
  document.querySelectorAll('[data-ablation]').forEach(button => button.addEventListener('click', () => {
    const index = Number(button.dataset.ablation), item = ablations[index];
    if (!item)
      return;
    $('ablation-images').replaceChildren(...item.images.map((path, i) => card(path, item.methods[i], item.prompt, i === 4)));
    document.querySelectorAll('[data-ablation]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    $('ablation-status').textContent = `Injection role comparison: ${item.prompt}.`;
  }));
  const box = $('lightbox');
  let trigger = null;
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-zoom]');
    if (!button || !box)
      return;
    trigger = button;
    const img = $('lightbox-image');
    img.src = button.dataset.zoom;
    img.alt = button.dataset.caption || '';
    img.classList.toggle('diagram', button.dataset.zoom.endsWith('.svg'));
    $('lightbox-title').textContent = button.dataset.caption || 'Figure';
    box.showModal();
  });
  $('lightbox-close')?.addEventListener('click', () => box.close());
  box?.addEventListener('close', () => trigger?.focus({ preventScroll: true }));
  box?.addEventListener('click', event => {
    if (event.target !== box)
      return;
    const r = box.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)
      box.close();
  });
})();

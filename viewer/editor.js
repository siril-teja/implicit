// Block canvas: an SVG editor for block graphs (see implicit/graph.py for the format).
// Drag blocks to move them, drag from an output socket to an input socket to wire
// them, drag a wire off an input to disconnect it. Double-click the background or
// press "+" to add a block. The selected block's settings are shown in the side panel.
// The editor only edits the graph; the viewer rebuilds the shader (Graph.build).
"use strict";

const Editor = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const W = 184, HEAD = 26, ROW = 20;
  const TYPE_COLORS = { body: "#e2b36d", bone: "#e8d6a0", scalar: "#7fd18b", vec3: "#6aa9ff" };

  function svgEl(tag, attrs = {}, parent = null) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.append(e);
    return e;
  }

  function h(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k in e) e[k] = v;
      else e.setAttribute(k, v);
    }
    for (const k of kids) if (k !== null && k !== undefined) e.append(k);
    return e;
  }

  const fmt = (v, step) => {
    const d = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step || 0.01)));
    return Number(v).toFixed(d);
  };

  function create(opts) {
    // opts: { svg, side, manifest, getGraph, getBuilt, getValue, onChange(kind), onValue(name, v),
    //         onSelect(sel), library }
    const svg = opts.svg;
    const blocks = Object.fromEntries(opts.manifest.blocks.map((b) => [b.name, b]));
    const view = { x: 40, y: 40, k: 0.85 };
    let selected = null;       // node id
    let focusBody = null;      // body index selected in 3D
    const root = svgEl("g", {}, svg);
    const wiresG = svgEl("g", {}, root);
    const nodesG = svgEl("g", {}, root);
    const tempWire = svgEl("path", { class: "wire temp", fill: "none" }, root);
    let geom = {};             // id -> { x, y, h, ins: {port: [x,y]}, outs: {port: [x,y]} }

    const graph = () => opts.getGraph();
    const nodes = () => graph().nodes;
    const applyView = () => root.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`);

    function toGraph(e) {
      const r = svg.getBoundingClientRect();
      return [(e.clientX - r.left - view.x) / view.k, (e.clientY - r.top - view.y) / view.k];
    }

    function inputsShown(blk) { return blk.inputs.filter((p) => Graph.WIRED.has(p.type)); }

    function literalText(n, port) {
      const v = (n.in || {})[port.name] ?? port.default;
      if (port.type === "scalar") return `${fmt(v, port.step)}${port.unit ? " " + port.unit : ""}`;
      if (port.type === "vec3" && Array.isArray(v)) return v.map((c) => fmt(c, port.step)).join(", ");
      return "";
    }

    // ------------------------------------------------------------ drawing
    function render() {
      const built = opts.getBuilt();
      const errors = built ? built.errors : {};
      nodesG.replaceChildren();
      geom = {};
      const hlNodes = highlightedNodes();
      for (const [id, n] of Object.entries(nodes())) {
        const blk = blocks[n.block];
        const [x, y] = n.pos || [0, 0];
        const ins = blk ? inputsShown(blk) : [];
        const outs = blk ? blk.outputs : [];
        const extra = n.block === "bone" || n.block === "output" || n.block === "number" ? 1 : 0;
        const rows = Math.max(ins.length, outs.length) + extra;
        const hgt = HEAD + Math.max(rows, 1) * ROW + 8;
        const g = svgEl("g", { class: "node" + (id === selected ? " sel" : "") + (errors[id] ? " err" : "")
          + (hlNodes.has(id) ? " hl" : ""), transform: `translate(${x},${y})` }, nodesG);
        g.dataset.id = id;
        svgEl("rect", { class: "box", width: W, height: hgt, rx: 6 }, g);
        svgEl("rect", { class: "head " + (blk ? "cat-" + blk.category.toLowerCase() : ""), width: W, height: HEAD, rx: 6 }, g);
        svgEl("rect", { class: "head-fill " + (blk ? "cat-" + blk.category.toLowerCase() : ""), y: HEAD - 6, width: W, height: 6 }, g);
        const t = svgEl("text", { class: "title", x: 9, y: 17 }, g);
        t.textContent = (n.title || id).slice(0, 24);
        const bt = svgEl("text", { class: "blk", x: W - 8, y: 17, "text-anchor": "end" }, g);
        bt.textContent = blk ? blk.title : n.block;
        if (errors[id]) {
          const tt = svgEl("title", {}, g);
          tt.textContent = errors[id];
        }
        const gm = { x, y, h: hgt, ins: {}, outs: {} };
        ins.forEach((p, i) => {
          const cy = HEAD + 14 + i * ROW;
          gm.ins[p.name] = [x, y + cy];
          svgEl("circle", { class: "sock in", cx: 0, cy, r: 5, fill: TYPE_COLORS[p.type], "data-port": p.name, "data-dir": "in" }, g);
          const lab = svgEl("text", { class: "port", x: 10, y: cy + 4 }, g);
          const wired = Graph.parseWire((n.in || {})[p.name]);
          const full = (p.label || p.name) + (wired ? "" : (literalText(n, p) ? "  " + literalText(n, p) : ""));
          const room = i < outs.length ? 22 : 30;  // leave space for an output label on the same row
          lab.textContent = full.length > room ? full.slice(0, room - 1) + "…" : full;
        });
        outs.forEach((p, i) => {
          const cy = HEAD + 14 + i * ROW;
          gm.outs[p.name] = [x + W, y + cy];
          svgEl("circle", { class: "sock out", cx: W, cy, r: 5, fill: TYPE_COLORS[p.type], "data-port": p.name, "data-dir": "out" }, g);
          const lab = svgEl("text", { class: "port", x: W - 10, y: cy + 4, "text-anchor": "end" }, g);
          lab.textContent = p.label || p.name;
        });
        if (extra) {
          const cy = HEAD + 14 + Math.max(ins.length, outs.length) * ROW;
          const lab = svgEl("text", { class: "port dim", x: 10, y: cy + 4 }, g);
          const i = n.in || {};
          lab.textContent = n.block === "bone" ? (libName(i.file) || "(no bone)")
            : n.block === "number" ? `= ${fmt(opts.getValue(id) ?? i.value ?? 0, i.step || 0.01)} ${i.unit || ""}`
              : `"${i.name || id}"`;
        }
        geom[id] = gm;
      }
      renderWires();
    }

    function libName(file) {
      const b = opts.library && opts.library.bones[file];
      return b ? b.name : file ? String(file).split("/").pop() : "";
    }

    function wirePath(a, b) {
      const dx = Math.max(40, Math.abs(b[0] - a[0]) * 0.5);
      return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
    }

    function renderWires() {
      wiresG.replaceChildren();
      for (const [id, n] of Object.entries(nodes())) {
        const blk = blocks[n.block];
        if (!blk) continue;
        for (const p of inputsShown(blk)) {
          const w = Graph.parseWire((n.in || {})[p.name]);
          if (!w || !geom[w.node] || !geom[id]) continue;
          const src = blocks[nodes()[w.node].block];
          const sp = w.port || (src && src.outputs[0] ? src.outputs[0].name : null);
          const a = geom[w.node].outs[sp], b = geom[id].ins[p.name];
          if (!a || !b) continue;
          const st = src && src.outputs.find((o) => o.name === sp);
          svgEl("path", { class: "wire", d: wirePath(a, b), stroke: TYPE_COLORS[st ? st.type : "body"] }, wiresG);
        }
      }
    }

    // ------------------------------------------------------------ interaction
    let act = null;  // { kind: "pan"|"move"|"wire", ... }
    svg.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      svg.setPointerCapture(e.pointerId);
      const sock = e.target.closest(".sock");
      const nodeEl = e.target.closest(".node");
      const [gx, gy] = toGraph(e);
      if (sock && nodeEl) {
        const id = nodeEl.dataset.id, port = sock.dataset.port;
        if (sock.dataset.dir === "out") {
          act = { kind: "wire", from: id, port, start: geom[id].outs[port] };
        } else {
          const cur = Graph.parseWire((nodes()[id].in || {})[port]);
          if (cur) {  // pick the wire up from its input
            const blk = blocks[nodes()[id].block];
            snapshot();
            nodes()[id].in[port] = defaultOf(blk, port);
            const src = blocks[nodes()[cur.node].block];
            const sp = cur.port || src.outputs[0].name;
            act = { kind: "wire", from: cur.node, port: sp, start: geom[cur.node].outs[sp], detached: true };
            opts.onChange("structure");
          }
        }
        return;
      }
      if (nodeEl) {
        const id = nodeEl.dataset.id;
        select(id);
        const p = nodes()[id].pos || [0, 0];
        act = { kind: "move", id, dx: gx - p[0], dy: gy - p[1], moved: false };
        return;
      }
      act = { kind: "pan", x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
    });

    svg.addEventListener("pointermove", (e) => {
      if (!act) return;
      const [gx, gy] = toGraph(e);
      if (act.kind === "pan") {
        view.x = act.vx + e.clientX - act.x;
        view.y = act.vy + e.clientY - act.y;
        act.moved = act.moved || Math.abs(e.clientX - act.x) + Math.abs(e.clientY - act.y) > 3;
        applyView();
      } else if (act.kind === "move") {
        if (!act.moved) snapshot();
        act.moved = true;
        nodes()[act.id].pos = [Math.round(gx - act.dx), Math.round(gy - act.dy)];
        render();
      } else if (act.kind === "wire") {
        tempWire.setAttribute("d", wirePath(act.start, [gx, gy]));
        const out = blocks[nodes()[act.from].block].outputs.find((o) => o.name === act.port);
        tempWire.setAttribute("stroke", TYPE_COLORS[out ? out.type : "body"]);
      }
    });

    svg.addEventListener("pointerup", (e) => {
      if (!act) return;
      const a = act;
      act = null;
      tempWire.setAttribute("d", "");
      if (a.kind === "pan" && !a.moved) select(null);
      if (a.kind === "move" && a.moved) opts.onChange("layout");
      if (a.kind === "wire") {
        const el = document.elementFromPoint(e.clientX, e.clientY);
        const sock = el && el.closest(".sock.in");
        const nodeEl = el && el.closest(".node");
        if (sock && nodeEl) connect(a.from, a.port, nodeEl.dataset.id, sock.dataset.port);
        else if (!a.detached) palette(e, { from: a.from, port: a.port });
      }
    });

    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const k = Math.min(2.5, Math.max(0.2, view.k * Math.exp(-e.deltaY * 0.0015)));
      view.x = mx - (mx - view.x) * (k / view.k);
      view.y = my - (my - view.y) * (k / view.k);
      view.k = k;
      applyView();
    }, { passive: false });

    svg.addEventListener("dblclick", (e) => {
      if (!e.target.closest(".node")) palette(e);
    });

    function defaultOf(blk, port) {
      const p = blk.inputs.find((q) => q.name === port);
      return p && p.default !== undefined ? JSON.parse(JSON.stringify(p.default)) : null;
    }

    function connect(from, fport, to, tport) {
      if (from === to) return;
      const src = blocks[nodes()[from].block], dst = blocks[nodes()[to].block];
      const op = src.outputs.find((o) => o.name === fport), ip = dst.inputs.find((i) => i.name === tport);
      if (!op || !ip || !(Graph.ACCEPTS[ip.type] || []).includes(op.type)) {
        opts.status(`Cannot connect ${op ? op.type : "?"} to ${ip ? ip.type : "?"} (${ip ? ip.label || ip.name : tport})`);
        return;
      }
      if (reaches(to, from)) { opts.status("That wire would make a loop"); return; }
      snapshot();
      nodes()[to].in = nodes()[to].in || {};
      nodes()[to].in[tport] = "@" + from + (fport === src.outputs[0].name ? "" : "." + fport);
      opts.onChange("structure");
    }

    function reaches(from, target) {  // does `from` feed (directly or not) into `target`?
      const up = (id, seen = new Set()) => {
        if (id === from) return true;
        if (seen.has(id)) return false;
        seen.add(id);
        return Object.values((nodes()[id] || {}).in || {}).some((v) => {
          const w = Graph.parseWire(v);
          return w && up(w.node, seen);
        });
      };
      return up(target);
    }

    // ------------------------------------------------------------ add / delete
    function newId(base) {
      base = base.replace(/[^A-Za-z0-9_]/g, "_").replace(/^([0-9])/, "_$1") || "node";
      let id = base, i = 2;
      while (nodes()[id]) id = `${base}_${i++}`;
      return id;
    }

    function addNode(blockName, pos, wireFrom) {
      const blk = blocks[blockName];
      snapshot();
      const id = newId(blockName);
      const n = { block: blockName, title: blk.title, pos: pos.map(Math.round), in: {} };
      if (blockName === "bone" && opts.library) n.in.file = Object.keys(opts.library.bones)[0];
      if (blockName === "output") n.in.name = id;
      nodes()[id] = n;
      if (wireFrom) {
        const src = blocks[nodes()[wireFrom.from].block];
        const op = src.outputs.find((o) => o.name === wireFrom.port);
        const ip = blk.inputs.find((p) => (Graph.ACCEPTS[p.type] || []).includes(op.type));
        if (ip) n.in[ip.name] = "@" + wireFrom.from + (wireFrom.port === src.outputs[0].name ? "" : "." + wireFrom.port);
      }
      selected = id;
      opts.onChange("structure");
      opts.onSelect({ node: id });
      return id;
    }

    function deleteNode(id) {
      if (!nodes()[id]) return;
      snapshot();
      delete nodes()[id];
      for (const n of Object.values(nodes())) {
        const blk = blocks[n.block];
        for (const [k, v] of Object.entries(n.in || {})) {
          const w = Graph.parseWire(v);
          if (w && w.node === id) n.in[k] = blk ? defaultOf(blk, k) : null;
        }
      }
      if (selected === id) select(null);
      opts.onChange("structure");
    }

    let pal = null;
    function palette(e, wireFrom = null) {
      closePalette();
      const pos = toGraph(e);
      let srcType = null;
      if (wireFrom) {
        const op = blocks[nodes()[wireFrom.from].block].outputs.find((o) => o.name === wireFrom.port);
        srcType = op && op.type;
      }
      const list = h("div", { className: "pal-list" });
      const search = h("input", { type: "search", placeholder: "Search blocks…", className: "pal-search" });
      const fill = () => {
        list.replaceChildren();
        const q = search.value.trim().toLowerCase();
        // Matches in the title first, then in the description; grouped by category when not searching.
        const score = (b) => {
          const title = (b.title + " " + b.name).toLowerCase();
          if (!q) return 0;
          if (title.startsWith(q)) return 1;
          if (title.includes(q)) return 2;
          return b.doc.toLowerCase().includes(q) ? 3 : -1;
        };
        const cats = {};
        const found = opts.manifest.blocks
          .filter((b) => !srcType || b.inputs.some((p) => (Graph.ACCEPTS[p.type] || []).includes(srcType)))
          .map((b) => [score(b), b]).filter(([s]) => s >= 0).sort((x, y) => x[0] - y[0]);
        for (const [, b] of found) {
          const cat = q ? "Matches" : b.category;
          (cats[cat] = cats[cat] || []).push(b);
        }
        for (const [cat, bs] of Object.entries(cats)) {
          list.append(h("div", { className: "pal-cat", textContent: cat }));
          for (const b of bs) {
            list.append(h("button", { className: "pal-item", title: b.doc,
              onclick: () => { closePalette(); addNode(b.name, pos, wireFrom); } },
            h("b", { textContent: b.title }), h("span", { textContent: b.doc })));
          }
        }
      };
      search.addEventListener("input", fill);
      search.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") closePalette();
        if (ev.key === "Enter") { const first = list.querySelector(".pal-item"); if (first) first.click(); }
      });
      fill();
      const host = svg.parentElement;
      const r = host.getBoundingClientRect();
      pal = h("div", { className: "palette" }, search, list);
      pal.style.left = Math.min(Math.max(0, e.clientX - r.left), r.width - 270) + "px";
      pal.style.top = Math.min(Math.max(0, e.clientY - r.top), Math.max(0, r.height - 320)) + "px";
      host.append(pal);
      search.focus();
    }
    function closePalette() { if (pal) { pal.remove(); pal = null; } }
    document.addEventListener("pointerdown", (e) => { if (pal && !pal.contains(e.target)) closePalette(); });

    // ------------------------------------------------------------ selection
    function highlightedNodes() {
      const built = opts.getBuilt();
      if (!built) return new Set();
      if (focusBody !== null) {
        const b = built.scene.bodies[focusBody];
        return b ? built.upstream(b.node) : new Set();
      }
      return new Set();
    }

    function select(id) {
      selected = id;
      focusBody = null;
      render();
      sidePanel();
      opts.onSelect(id ? { node: id } : null);
    }

    function selectBody(index) {
      selected = null;
      focusBody = index;
      render();
      sidePanel();
      const built = opts.getBuilt();
      const b = index !== null && built && built.scene.bodies[index];
      if (b) reveal(built.upstream(b.node));
    }

    function reveal(ids) {  // pan so the given nodes are in view
      const gs = [...ids].map((i) => geom[i]).filter(Boolean);
      if (!gs.length) return;
      const r = svg.getBoundingClientRect();
      const x0 = Math.min(...gs.map((g) => g.x)), x1 = Math.max(...gs.map((g) => g.x + W));
      const y0 = Math.min(...gs.map((g) => g.y)), y1 = Math.max(...gs.map((g) => g.y + g.h));
      const inView = x0 * view.k + view.x >= 0 && x1 * view.k + view.x <= r.width
        && y0 * view.k + view.y >= 0 && y1 * view.k + view.y <= r.height;
      if (inView) return;
      view.x = r.width / 2 - ((x0 + x1) / 2) * view.k;
      view.y = r.height / 2 - ((y0 + y1) / 2) * view.k;
      applyView();
    }

    function fit() {
      const gs = Object.values(geom);
      const r = svg.getBoundingClientRect();
      if (!gs.length || !r.width) return;
      const x0 = Math.min(...gs.map((g) => g.x)) - 30, x1 = Math.max(...gs.map((g) => g.x + W)) + 30;
      const y0 = Math.min(...gs.map((g) => g.y)) - 30, y1 = Math.max(...gs.map((g) => g.y + g.h)) + 30;
      view.k = Math.min(1.2, Math.max(0.2, Math.min(r.width / (x1 - x0), r.height / (y1 - y0))));
      view.x = (r.width - (x1 - x0) * view.k) / 2 - x0 * view.k;
      view.y = (r.height - (y1 - y0) * view.k) / 2 - y0 * view.k;
      applyView();
    }

    // ------------------------------------------------------------ side panel
    function sidePanel() {
      const side = opts.side;
      side.replaceChildren();
      const built = opts.getBuilt();
      if (focusBody !== null && built && built.scene.bodies[focusBody]) {
        const b = built.scene.bodies[focusBody];
        side.append(h("h1", { textContent: b.name }),
          h("div", { className: "hint", textContent: "Every block that builds this body, from its output back to its inputs. Click a block on the canvas for just that block." }));
        const chain = [...built.upstream(b.node)].reverse();  // inputs first, output last
        for (const id of chain) if (nodes()[id]) side.append(nodeSection(id, true));
        return;
      }
      if (!selected || !nodes()[selected]) {
        side.append(h("div", { className: "hint", textContent:
          "Click a block to edit it, or click a body in the 3D view. Double-click the canvas (or +) to add a block; drag from an output socket to an input socket to connect." }));
        return;
      }
      side.append(nodeSection(selected, false));
    }

    function nodeSection(id, compact) {
      const n = nodes()[id], blk = blocks[n.block];
      const built = opts.getBuilt();
      const box = h("div", { className: "nsec" });
      const title = h("input", { className: "ntitle", value: n.title || id, onchange: (e) => {
        snapshot(); n.title = e.target.value; opts.onChange("structure");
      } });
      box.append(h("div", { className: "nhead" }, title,
        h("span", { className: "nblk", textContent: blk ? blk.title : n.block }),
        compact ? h("button", { className: "reset", textContent: "→", title: "Show this block", onclick: () => select(id) }) : null));
      if (!compact && blk) box.append(h("div", { className: "hint", textContent: blk.doc }));
      if (built && built.errors[id]) box.append(h("div", { className: "nerr", textContent: built.errors[id] }));
      if (!blk) return box;
      for (const p of blk.inputs) {
        const row = inputEditor(id, n, blk, p);
        if (row) box.append(row);
      }
      if (!compact) {
        box.append(h("div", { className: "buttons", style: "margin-top:10px" },
          h("button", { textContent: "Delete block", onclick: () => deleteNode(id) })));
      }
      return box;
    }

    function inputEditor(id, n, blk, p) {
      n.in = n.in || {};
      const v = p.name in n.in ? n.in[p.name] : p.default;
      const label = h("label", { textContent: (p.label || p.name) + (p.unit ? ` (${p.unit})` : "") });
      const wire = Graph.parseWire(v);
      if (wire) {
        const src = nodes()[wire.node];
        return h("div", { className: "param" }, label, h("div", { className: "row" },
          h("span", { className: "wired", textContent: "← " + (src ? src.title || wire.node : wire.node) }),
          Graph.WIRED.has(p.type) && p.type !== "body" && p.type !== "bone"
            ? h("button", { className: "reset", textContent: "✕", title: "Disconnect", onclick: () => {
              snapshot(); n.in[p.name] = defaultOf(blk, p.name); opts.onChange("structure");
            } }) : null));
      }
      if (p.type === "body" || p.type === "bone") {
        return h("div", { className: "param" }, label, h("div", { className: "hint", textContent: `Not connected: wire a ${p.type} here.` }));
      }
      const setStatic = (val) => { snapshot(); n.in[p.name] = val; opts.onChange("structure"); };
      if (p.type === "scalar") return slider(label, `${id}_${p.name}`, v, p, (val) => { n.in[p.name] = val; });
      if (p.type === "vec3") {
        const arr = Array.isArray(v) ? v.slice() : [0, 0, 0];
        const wrap = h("div", { className: "param" }, label);
        ["x", "y", "z"].forEach((a, i) => {
          wrap.append(slider(h("label", { className: "sub", textContent: a }), `${id}_${p.name}_${a}`, arr[i], p, (val) => {
            arr[i] = val; n.in[p.name] = arr.slice();
          }, true));
        });
        return wrap;
      }
      if (n.block === "number" && p.name === "value") {
        const q = { ...p, lo: n.in.min ?? 0, hi: n.in.max ?? 1, step: n.in.step || 0.01 };
        return slider(label, id, v, q, (val) => { n.in.value = val; });
      }
      if (p.type === "number") {
        return h("div", { className: "param" }, label, h("input", { type: "number", value: v, step: "any",
          onchange: (e) => setStatic(parseFloat(e.target.value)) }));
      }
      if (p.type === "choice") {
        return h("div", { className: "param" }, label, h("select", { onchange: (e) => setStatic(e.target.value) },
          ...p.choices.map((c) => h("option", { value: c, textContent: c, selected: c === v }))));
      }
      if (p.type === "bool") {
        return h("label", { className: "body" }, h("input", { type: "checkbox", checked: !!v, onchange: (e) => setStatic(e.target.checked) }),
          p.label || p.name);
      }
      if (p.type === "color") {
        const hex = Array.isArray(v) ? "#" + v.map((c) => Math.round(c * 255).toString(16).padStart(2, "0")).join("") : v;
        return h("div", { className: "param" }, label, h("input", { type: "color", value: hex, onchange: (e) => setStatic(e.target.value) }));
      }
      if (p.type === "path") {
        const bones = opts.library ? Object.entries(opts.library.bones) : [];
        return h("div", { className: "param" }, label, h("select", { onchange: (e) => setStatic(e.target.value) },
          ...bones.map(([key, b]) => h("option", { value: key, textContent: `${b.name} (${b.kind})`, selected: key === v }))));
      }
      return h("div", { className: "param" }, label, h("input", { type: "text", value: v ?? "",
        onchange: (e) => setStatic(e.target.value) }));
    }

    // Live slider for a parameter: moves the uniform at once, records the value in the graph.
    function slider(label, pname, v, p, store, compact = false) {
      const lo = Math.min(p.lo ?? v, v), hi = Math.max(p.hi ?? v, v);
      const step = p.step || (hi - lo) / 200;
      const cur = opts.getValue(pname) ?? v;
      const num = h("input", { type: "number", className: "num", value: fmt(cur, step), step: "any" });
      const range = h("input", { type: "range", min: lo, max: hi, step, value: cur });
      let snap = false;
      const set = (val, commit) => {
        if (!snap) { snapshot(); snap = true; }
        store(val);
        opts.onValue(pname, val);
        if (commit) { snap = false; opts.onChange(val < lo || val > hi ? "structure" : "value"); }
      };
      range.addEventListener("input", (e) => { const val = parseFloat(e.target.value); num.value = fmt(val, step); set(val, false); });
      range.addEventListener("change", (e) => set(parseFloat(e.target.value), true));
      num.addEventListener("change", (e) => {
        const val = parseFloat(e.target.value);
        if (!Number.isFinite(val)) return;
        range.value = val;
        set(val, true);
      });
      return h("div", { className: compact ? "param sub" : "param" }, label, h("div", { className: "row" }, range, num));
    }

    // ------------------------------------------------------------ history
    const undoStack = [], redoStack = [];
    function snapshot() {
      undoStack.push(JSON.stringify(graph()));
      if (undoStack.length > 200) undoStack.shift();
      redoStack.length = 0;
    }
    function undo() {
      if (!undoStack.length) return;
      redoStack.push(JSON.stringify(graph()));
      opts.replaceGraph(JSON.parse(undoStack.pop()));
    }
    function redo() {
      if (!redoStack.length) return;
      undoStack.push(JSON.stringify(graph()));
      opts.replaceGraph(JSON.parse(redoStack.pop()));
    }

    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement && document.activeElement.tagName);
      if (typing || !svg.isConnected || svg.closest("[hidden]")) return;
      if ((e.key === "Delete" || e.key === "Backspace") && selected) { e.preventDefault(); deleteNode(selected); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); redo(); }
    });

    applyView();
    return {
      render, sidePanel, select, selectBody, fit, undo, redo, snapshot,
      addAtCenter(e) {
        const r = svg.getBoundingClientRect();
        palette({ clientX: r.left + r.width / 2 - 130, clientY: r.top + 30 });
        void e;
      },
      get selected() { return selected; },
      get focusBody() { return focusBody; },
    };
  }

  return { create };
})();

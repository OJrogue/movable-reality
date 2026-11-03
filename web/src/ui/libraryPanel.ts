// Lane ui: the panel behind the top bar's + button. "Add" lists the premade meshes (library/index.ts);
// a click stands one in the room and selects it, ready to drag. "In this room" lists what the
// room holds, scanned or added, with a way to remove or bring back each item. Removing never
// deletes: it hides the object, as the Delete key and the agent do (scene.py).
import type { SceneManifest, SceneObject } from "../scene";
import { byCategory, LIBRARY_BASE, type LibraryItem, loadCatalog } from "../library";
import type { HistoryEntry } from "./edits";

export interface LibraryPanelDeps {
  add(item: LibraryItem): void;
  remove(id: string): void;
  restore(id: string): void;
  select(id: string): void;
  /** Phones: the panel covers the room, so it closes after adding and yields to the help card. */
  touch(): boolean;
  closeOthers(): void;
}

export interface RoomRow {
  object: SceneObject;
  removed: boolean;
}

/** What the "In this room" list shows: every visible object and every one the user removed.
 * Other hidden objects stay out: the pipeline hides merged copies (objtrain.install), and an
 * undone insert was never meant to be there. */
export function roomRows(m: Pick<SceneManifest, "objects" | "history">): RoomRow[] {
  const removed = new Set((m.history as HistoryEntry[]).filter((e) => e.op === "remove").map((e) => e.object_id));
  return m.objects
    .filter((o) => o.visible || removed.has(o.id))
    .map((object) => ({ object, removed: !object.visible }))
    .sort((a, b) => Number(a.removed) - Number(b.removed) || a.object.label.localeCompare(b.object.label));
}

const SOURCE: Record<SceneObject["source"], string> = { scan: "Scanned", generated: "Generated", library: "Added" };
const PLUS =
  '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>';

const sizeText = (s: LibraryItem["size"]) => `${s[0].toFixed(1)} × ${s[2].toFixed(1)} m`;

function div(cls: string, text = ""): HTMLDivElement {
  const e = document.createElement("div");
  e.className = cls;
  e.textContent = text;
  return e;
}

export interface LibraryPanelHandle {
  button: HTMLButtonElement;
  setManifest(m: SceneManifest): void;
  shown(): boolean;
  show(on: boolean): void;
}

export function installLibraryPanel(deps: LibraryPanelDeps): LibraryPanelHandle {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "mr-chrome-icon";
  button.innerHTML = PLUS;
  button.title = "Add furniture, or remove what is in the room";
  button.setAttribute("aria-label", "Add furniture, or remove what is in the room");
  button.setAttribute("aria-expanded", "false");

  const panel = div("mr-ui-lib");
  panel.hidden = true;
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Furniture");
  // Keys inside the panel must not walk the camera or trigger the viewer's shortcuts.
  panel.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Escape") show(false);
  });

  const tabs = div("mr-ui-lib-tabs");
  const tab = (text: string) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    tabs.append(b);
    return b;
  };
  const addTab = tab("Add");
  const roomTab = tab("In this room");
  const body = div("mr-ui-lib-body");
  panel.append(tabs, body);
  document.body.append(panel);

  let catalog: LibraryItem[] = [];
  let manifest: SceneManifest | null = null;
  let page: "add" | "room" = "add";
  let roomKey = "";

  const renderAdd = () => {
    body.replaceChildren();
    if (!catalog.length) {
      body.append(div("mr-ui-lib-note", "The furniture library is not available here."));
      return;
    }
    for (const [category, items] of byCategory(catalog)) {
      body.append(div("mr-ui-lib-heading", category));
      const grid = div("mr-ui-lib-grid");
      for (const item of items) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "mr-ui-lib-card";
        card.title = item.on_surface
          ? `Add a ${item.label}: look at a table or shelf to stand it there`
          : `Add a ${item.label} where you are looking`;
        if (item.thumb) {
          const img = document.createElement("img");
          img.src = `${LIBRARY_BASE}/${item.thumb}`;
          img.alt = "";
          img.loading = "lazy";
          card.append(img);
        }
        card.append(div("mr-ui-lib-name", item.label), div("mr-ui-lib-size", sizeText(item.size)));
        card.addEventListener("click", () => {
          deps.add(item);
          if (deps.touch()) show(false);
        });
        grid.append(card);
      }
      body.append(grid);
    }
  };

  const renderRoom = () => {
    body.replaceChildren();
    const rows = manifest ? roomRows(manifest) : [];
    if (!rows.length) body.append(div("mr-ui-lib-note", "Nothing in this room yet."));
    for (const { object, removed } of rows) {
      const row = div("mr-ui-lib-row");
      row.dataset.removed = removed ? "1" : "0";
      const name = document.createElement("button");
      name.type = "button";
      name.className = "mr-ui-lib-rowname";
      name.disabled = removed;
      name.title = removed ? "" : "Select it";
      name.append(div("mr-ui-lib-name", object.label), div("mr-ui-lib-size", removed ? "Removed" : SOURCE[object.source]));
      name.addEventListener("click", () => {
        deps.select(object.id);
        if (deps.touch()) show(false);
      });
      const action = document.createElement("button");
      action.type = "button";
      action.className = "mr-ui-lib-action";
      action.textContent = removed ? "Restore" : "Remove";
      action.setAttribute("aria-label", `${removed ? "Restore" : "Remove"} the ${object.label}`);
      action.addEventListener("click", () => (removed ? deps.restore(object.id) : deps.remove(object.id)));
      row.append(name, action);
      body.append(row);
    }
  };

  const render = () => {
    addTab.classList.toggle("on", page === "add");
    roomTab.classList.toggle("on", page === "room");
    if (page === "add") renderAdd();
    else renderRoom();
  };

  addTab.addEventListener("click", () => {
    page = "add";
    render();
  });
  roomTab.addEventListener("click", () => {
    page = "room";
    render();
  });

  const show = (on: boolean) => {
    panel.hidden = !on;
    button.setAttribute("aria-expanded", String(on));
    if (on) {
      if (deps.touch()) deps.closeOthers();
      render();
    }
    button.blur(); // Space must stay hold-to-talk
  };
  button.addEventListener("click", () => show(panel.hidden));

  void loadCatalog().then((items) => {
    catalog = items;
    if (!panel.hidden && page === "add") render();
  });

  return {
    button,
    shown: () => !panel.hidden,
    show,
    setManifest(m: SceneManifest) {
      manifest = m;
      const n = m.objects.filter((o) => o.visible).length;
      roomTab.textContent = `In this room (${n})`;
      // Only a change to the list redraws it: the manifest also changes on every drag.
      const key = roomRows(m).map((r) => `${r.object.id}:${r.removed}`).join();
      if (key !== roomKey && page === "room" && !panel.hidden) render();
      roomKey = key;
    },
  };
}

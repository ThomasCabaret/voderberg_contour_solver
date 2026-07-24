#!/usr/bin/env python3
"""Local web viewer for retained Voderberg contour solver results.

The server uses only the Python standard library. It streams the selected JSON
file to the browser instead of duplicating it in server memory. The interface
renders only one page of rows at a time.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import settings


DEFAULT_HOST = settings.DEFAULT_WEB_HOST
DEFAULT_PORT = settings.DEFAULT_WEB_PORT
DEFAULT_RESULTS_FILE = Path(settings.DEFAULT_WEB_RESULTS_FILE)


HTML_PAGE = r'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Voderberg contour results</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0d1117;
      --panel: #161b22;
      --panel2: #1f2630;
      --text: #e6edf3;
      --muted: #9da7b3;
      --border: #30363d;
      --good: #3fb950;
      --bad: #f85149;
      --warn: #d29922;
      --accent: #58a6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    header { padding: 24px 28px 12px; }
    h1 { margin: 0 0 6px; font-size: 25px; }
    .subtitle { color: var(--muted); font-size: 13px; }
    main { padding: 12px 28px 32px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 13px 15px;
    }
    .card .label { color: var(--muted); font-size: 12px; }
    .card .value { margin-top: 4px; font-size: 23px; font-weight: 700; }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(5, minmax(120px, 1fr));
      gap: 9px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 12px;
    }
    input, select, button {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--panel2);
      color: var(--text);
      padding: 9px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    .table-wrap {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: auto;
      background: var(--panel);
      max-height: calc(100vh - 285px);
    }
    .pager {
      display: grid;
      grid-template-columns: auto minmax(180px, 1fr) auto auto;
      align-items: center;
      gap: 9px;
      margin-top: 10px;
    }
    .pager button, .pager select { width: auto; }
    .pager .page-info { color: var(--muted); text-align: center; }
    table { border-collapse: collapse; width: 100%; min-width: 1220px; }
    th, td {
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 9px 10px;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--panel2);
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: rgba(88,166,255,.08); }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge.good { background: rgba(63,185,80,.16); color: #75d486; }
    .badge.bad { background: rgba(248,81,73,.16); color: #ff7b72; }
    .badge.warn { background: rgba(210,153,34,.18); color: #e3b341; }
    .badge.info { background: rgba(88,166,255,.16); color: #79c0ff; }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    .mapping { max-width: 360px; line-height: 1.35; }
    .mapping div + div { margin-top: 5px; }
    .profile { min-width: 430px; max-width: 680px; line-height: 1.5; overflow-wrap: anywhere; }
    .muted { color: var(--muted); }
    .empty { padding: 30px; color: var(--muted); text-align: center; }
    dialog {
      width: min(1040px, calc(100vw - 32px));
      max-height: calc(100vh - 32px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0;
      background: var(--panel);
      color: var(--text);
    }
    dialog::backdrop { background: rgba(0,0,0,.72); }
    .dialog-head {
      position: sticky; top: 0; z-index: 2;
      display: flex; justify-content: space-between; gap: 12px; align-items: center;
      background: var(--panel2); border-bottom: 1px solid var(--border); padding: 14px 16px;
    }
    .dialog-head h2 { margin: 0; font-size: 18px; }
    .dialog-head button { width: auto; padding: 7px 12px; }
    .dialog-body { padding: 16px; overflow: auto; }
    .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .detail-section {
      background: var(--bg); border: 1px solid var(--border); border-radius: 9px; padding: 12px;
    }
    .detail-section.full { grid-column: 1 / -1; }
    .detail-section h3 { margin: 0 0 9px; font-size: 14px; }
    dl { display: grid; grid-template-columns: 150px 1fr; gap: 6px 10px; margin: 0; font-size: 13px; }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; }
    pre {
      margin: 0; padding: 10px; overflow: auto; background: #090c10;
      border: 1px solid var(--border); border-radius: 7px; font-size: 12px;
    }
    @media (max-width: 1000px) {
      .controls { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .controls input { grid-column: 1 / -1; }
      .detail-grid { grid-template-columns: 1fr; }
      .detail-section.full { grid-column: auto; }
    }
  </style>
</head>
<body>
<header>
  <h1>Resultats formels des contours</h1>
  <div class="subtitle" id="sourceLine">Chargement...</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="controls">
    <input id="search" type="search" placeholder="Rechercher profil, mapping, equation, case...">
    <select id="statusFilter">
      <option value="all">Tous les statuts</option>
      <option value="retained">Retenus</option>
      <option value="rejected">Rejetes</option>
      <option value="total_turn">Rejet: tour total</option>
      <option value="pole_angles">Rejet: poles</option>
      <option value="total_turn_and_poles">Rejet: tour + poles</option>
      <option value="translation">Rejet: translation</option>
    </select>
    <select id="parityFilter"><option value="all">Toutes parites</option></select>
    <select id="flipFilter">
      <option value="all">Tous les flips</option>
      <option value="none">Aucun flip</option>
      <option value="one">Un flip</option>
      <option value="two">Deux flips</option>
    </select>
    <select id="sortKey">
      <option value="importance">Tri: importance</option>
      <option value="case">Tri: case</option>
      <option value="complexity">Tri: complexite</option>
      <option value="parameters">Tri: parametres</option>
      <option value="parity">Tri: parite</option>
    </select>
    <select id="sortDirection">
      <option value="asc">Croissant</option>
      <option value="desc">Decroissant</option>
    </select>
  </section>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Statut</th><th>Case / profil</th><th>Parite</th><th>Appariement retenu</th>
        <th>Profil formel complet</th><th>Critere determinant</th><th>Taille</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
    <div id="empty" class="empty" hidden>Aucun profil ne correspond aux filtres.</div>
  </div>
  <section class="pager">
    <button id="previousPage">Page precedente</button>
    <div class="page-info" id="pageInfo"></div>
    <select id="pageSize" aria-label="Nombre de profils par page"></select>
    <button id="nextPage">Page suivante</button>
  </section>
</main>
<dialog id="detailDialog">
  <div class="dialog-head"><h2 id="detailTitle"></h2><button id="closeDialog">Fermer</button></div>
  <div class="dialog-body" id="detailBody"></div>
</dialog>
<script>
'use strict';

const PAGE_SIZE_DEFAULT = __WEB_DEFAULT_PAGE_SIZE__;
const PAGE_SIZE_OPTIONS = __WEB_PAGE_SIZE_OPTIONS__;
const state = {
  payload: null,
  profiles: [],
  visible: [],
  page: 1,
  pageSize: PAGE_SIZE_DEFAULT,
};
const stageLabels = {
  retained: 'Retenu', translation: 'Translation', pole_angles: 'Angles aux poles',
  total_turn: 'Tour total', total_turn_and_poles: 'Tour + poles'
};

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function badge(text, kind) { return `<span class="badge ${kind}">${esc(text)}</span>`; }
function statusBadge(profile) {
  const core = profile.status.retained ? badge('Retenu core', 'good') : badge(stageLabels[profile.status.stage] || profile.status.stage, 'bad');
  const experimental = profile.experimental?.exact_encoded_model_rejection ? ` ${badge('Rejet exp.', 'warn')}` : '';
  return core + experimental;
}
function flipCount(profile) { return Number(profile.mapping.A.flipped) + Number(profile.mapping.B.flipped); }
function normalizedSearch(profile) {
  return [profile.case_id, profile.profile_id, profile.solution.profile, profile.solution.word_contour, profile.solution.contour, profile.mapping.A.display, profile.mapping.B.display,
    ...(profile.placement.equations || []), ...(profile.status.reasons || []), profile.experimental?.status || '', profile.experimental?.reason || '', profile.solution_equivalence?.key || ''].join(' ').toLowerCase();
}

function renderCards() {
  const profiles = state.profiles;
  const retained = profiles.filter(p => p.status.retained).length;
  const stages = Object.fromEntries(['total_turn','pole_angles','total_turn_and_poles','translation'].map(s => [s, profiles.filter(p => p.status.stage === s).length]));
  const expRejects = profiles.filter(p => p.experimental?.exact_encoded_model_rejection).length;
  const expAdditional = profiles.filter(p => p.status.retained && p.experimental?.exact_encoded_model_rejection).length;
  const values = [
    ['Profils affichables', profiles.length], ['Retenus core', retained],
    ['Rejet angle global', stages.total_turn + stages.total_turn_and_poles],
    ['Rejet poles', stages.pole_angles + stages.total_turn_and_poles],
    ['Rejet translation core', stages.translation],
    ['Rejets exp. encodes', expRejects],
    ['Rejets exp. additionnels', expAdditional]
  ];
  document.getElementById('cards').innerHTML = values.map(([label,value]) => `<div class="card"><div class="label">${esc(label)}</div><div class="value">${value}</div></div>`).join('');
}

function populateParity() {
  const select = document.getElementById('parityFilter');
  [...new Set(state.profiles.map(p => p.placement.contact_parity))].sort().forEach(parity => {
    const option = document.createElement('option'); option.value = parity; option.textContent = `Parite ${parity}`; select.appendChild(option);
  });
}

function compareProfiles(a, b, key) {
  if (key === 'case') return (a.case_id - b.case_id) || (a.profile_id - b.profile_id);
  if (key === 'complexity') return (a.solution.word_token_count - b.solution.word_token_count) || (a.case_id - b.case_id);
  if (key === 'parameters') return (a.solution.parameter_count - b.solution.parameter_count) || (a.solution.word_token_count - b.solution.word_token_count);
  if (key === 'parity') return a.placement.contact_parity.localeCompare(b.placement.contact_parity) || (a.case_id - b.case_id);
  return (a.sort.retained_priority - b.sort.retained_priority)
    || (a.sort.stage_priority - b.sort.stage_priority)
    || (a.sort.reflection_count - b.sort.reflection_count)
    || (a.sort.word_token_count - b.sort.word_token_count)
    || (a.case_id - b.case_id);
}

function applyFilters() {
  const query = document.getElementById('search').value.trim().toLowerCase();
  const status = document.getElementById('statusFilter').value;
  const parity = document.getElementById('parityFilter').value;
  const flip = document.getElementById('flipFilter').value;
  const sortKey = document.getElementById('sortKey').value;
  const direction = document.getElementById('sortDirection').value === 'desc' ? -1 : 1;

  state.visible = state.profiles.filter(profile => {
    if (query && !normalizedSearch(profile).includes(query)) return false;
    if (status === 'retained' && !profile.status.retained) return false;
    if (status === 'rejected' && profile.status.retained) return false;
    if (!['all','retained','rejected'].includes(status) && profile.status.stage !== status) return false;
    if (parity !== 'all' && profile.placement.contact_parity !== parity) return false;
    const count = flipCount(profile);
    if (flip === 'none' && count !== 0) return false;
    if (flip === 'one' && count !== 1) return false;
    if (flip === 'two' && count !== 2) return false;
    return true;
  }).sort((a,b) => direction * compareProfiles(a,b,sortKey));
  state.page = 1;
  renderRows();
}

function renderRows() {
  const body = document.getElementById('rows');
  const empty = document.getElementById('empty');
  empty.hidden = state.visible.length !== 0;
  const pageCount = Math.max(1, Math.ceil(state.visible.length / state.pageSize));
  state.page = Math.min(Math.max(1, state.page), pageCount);
  const start = (state.page - 1) * state.pageSize;
  const pageProfiles = state.visible.slice(start, start + state.pageSize);
  body.innerHTML = pageProfiles.map(profile => {
    const reason = profile.experimental?.exact_encoded_model_rejection ? `Couche experimentale: ${profile.experimental.status}` : (profile.status.retained ? 'Tous les filtres core passent' : (profile.status.reasons[0] || stageLabels[profile.status.stage]));
    return `<tr data-id="${profile.profile_id}">
      <td>${statusBadge(profile)}</td>
      <td><strong>#${profile.case_id}</strong><br><span class="muted">profil ${profile.profile_id}</span>${profile.solution_equivalence?.key ? `<br><span class="muted">classe ${profile.solution_equivalence.key.slice(0, 10)}… (${profile.solution_equivalence.class_size_within_bounded_terminal_output || 1})</span>` : ''}</td>
      <td>${badge(profile.placement.contact_parity, 'info')}<br><span class="muted">A ${profile.mapping.A.flipped ? 'flip' : 'direct'}; B ${profile.mapping.B.flipped ? 'flip' : 'direct'}</span></td>
      <td class="mapping"><div><strong>A:</strong> ${esc(profile.mapping.A.display)}</div><div><strong>B:</strong> ${esc(profile.mapping.B.display)}</div></td>
      <td class="profile mono">${esc(profile.solution.profile)}</td>
      <td>${esc(reason)}</td>
      <td>${profile.solution.word_token_count} mots<br><span class="muted">${profile.solution.parameter_count} param.; profondeur ${profile.solution.solver_depth}</span></td>
    </tr>`;
  }).join('');
  const end = Math.min(start + pageProfiles.length, state.visible.length);
  document.getElementById('pageInfo').textContent = state.visible.length
    ? `${start + 1}-${end} sur ${state.visible.length} profils filtres (page ${state.page}/${pageCount})`
    : '0 profil filtre';
  document.getElementById('previousPage').disabled = state.page <= 1;
  document.getElementById('nextPage').disabled = state.page >= pageCount;
}

function detailSection(title, content, full=false) {
  return `<section class="detail-section${full ? ' full' : ''}"><h3>${esc(title)}</h3>${content}</section>`;
}
function dl(items) {
  return `<dl>${items.map(([k,v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
}
function showDetail(id) {
  const p = state.profiles.find(item => item.profile_id === id);
  if (!p) return;
  document.getElementById('detailTitle').innerHTML = `Case ${p.case_id}, profil ${p.profile_id} &nbsp; ${statusBadge(p)}`;
  const mapping = dl([
    ['Copie A', esc(p.mapping.A.display)], ['Copie A flip', p.mapping.A.flipped ? 'oui' : 'non'],
    ['Copie B', esc(p.mapping.B.display)], ['Copie B flip', p.mapping.B.flipped ? 'oui' : 'non'],
    ['Parite', esc(p.placement.contact_parity)], ['Equations', `<span class="mono">${p.placement.equations.map(esc).join('<br>')}</span>`]
  ]);
  const angleClasses = (p.solution.formal_profile?.angle_classes || []).map(item => {
    const members = item.members.map(member => `${member.sign < 0 ? '-' : ''}${member.point}`).join(', ');
    return `<span class="mono">${esc(item.alias)}${item.fixed_zero ? ' = 0' : ''}</span>: ${esc(members)}`;
  }).join('<br>') || '(indisponible dans cet ancien export)';
  const solution = dl([
    ['Profil complet', `<span class="mono">${esc(p.solution.profile)}</span>`],
    ['Mots seuls', `<span class="mono">${esc(p.solution.word_contour)}</span>`],
    ['Classes d angles', angleClasses],
    ['Parametres courbes', esc((p.solution.curve_parameters || []).join(', '))],
    ['Parametres angles', esc((p.solution.angle_parameters || []).join(', '))],
    ['Angles forces droits', esc((p.solution.fixed_zero_angle_classes || []).join(', ') || 'aucun')],
    ['Derivation', esc(p.solution.derivation.join(' -> ') || '(terminale directement)')],
    ['Profondeur', p.solution.solver_depth]
  ]);
  const equivalence = p.solution_equivalence?.key ? dl([
    ['Cle decoree', `<span class="mono">${esc(p.solution_equivalence.key)}</span>`],
    ['Taille de classe bornee', p.solution_equivalence.class_size_within_bounded_terminal_output || 1],
    ['Profil representant', p.solution_equivalence.representative_profile_id ?? p.profile_id],
    ['Mapping inclus', p.solution_equivalence.copy_mapping_included ? 'oui' : 'non'],
    ['Miroir global identifie', p.solution_equivalence.global_mirror_identified ? 'oui' : 'non'],
    ['Cycles parametriques reconnus', p.solution_equivalence.parametric_cycle_families_identified ? 'oui' : 'non']
  ]) : '<p>Canonicalisation des solutions desactivee ou indisponible.</p>';
  const filterSummary = dl([
    ['Tour total', p.filters.total_turn.feasible ? badge('passe','good') : badge('rejete','bad')],
    ['Equation de tour', `<span class="mono">${esc(p.filters.total_turn.equation)}</span>`],
    ['Angles aux poles', p.filters.pole_angles.feasible ? badge('passe','good') : badge('rejete','bad')],
    ['Capacite conjointe', esc(p.filters.pole_angles.joint_capacity_pi_units?.decimal)],
    ['Translation', p.filters.translation_pass ? badge('non contredite','good') : badge('rejete','bad')],
    ['Statut translation', esc(p.filters.translation_holonomy.status)],
    ['Raisons', (p.status.reasons || []).map(esc).join('<br>') || 'Aucune contradiction trouvee']
  ]);
  const experimental = p.experimental || {};
  const external = experimental.external_boundary || {};
  const z3 = experimental.z3_result || {};
  const experimentalSummary = dl([
    ['Statut experimental', esc(experimental.status || 'non demande')],
    ['Affecte le statut principal', experimental.affects_core_status ? 'oui' : 'non'],
    ['Contradiction exacte du modele encode', experimental.exact_encoded_model_rejection ? badge('oui','bad') : badge('non','good')],
    ['Rotation conjointe', external.joint_rotation_analysis ? (external.joint_rotation_analysis.feasible ? badge('passe','good') : badge('rejete','bad')) : 'non calculee'],
    ['Translation conjointe elementaire', external.joint_translation_analysis ? esc(external.joint_translation_analysis.status) : 'non calculee'],
    ['Z3/NLSAT', z3.status ? esc(z3.status) : (experimental.z3_problem ? 'probleme genere, non execute' : 'non prepare')],
    ['Portee', esc(experimental.model_scope || '')]
  ]);
  const raw = `<pre>${esc(JSON.stringify(p, null, 2))}</pre>`;
  document.getElementById('detailBody').innerHTML = `<div class="detail-grid">
    ${detailSection('Appariement', mapping)}${detailSection('Profil solution', solution)}
    ${detailSection('Equivalence de solution', equivalence, true)}${detailSection('Filtres principaux', filterSummary, true)}${detailSection('Couche experimentale', experimentalSummary, true)}${detailSection('JSON complet', raw, true)}
  </div>`;
  document.getElementById('detailDialog').showModal();
}

async function init() {
  const response = await fetch('/api/data');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  state.payload = await response.json();
  state.profiles = state.payload.profiles || [];
  const source = response.headers.get('X-Source-File') || state.payload.metadata?.source_file || '';
  document.getElementById('sourceLine').textContent = `schema ${state.payload.metadata?.schema_version || '?'} - ${source} - ${state.profiles.length} profils retenus. Audit borne: profondeur ${state.payload.metadata?.max_solver_depth_per_case ?? '?'}, ${state.payload.metadata?.max_solver_states_per_case ?? '?'} etats/case.`;
  const pageSize = document.getElementById('pageSize');
  PAGE_SIZE_OPTIONS.forEach(value => {
    const option = document.createElement('option');
    option.value = value; option.textContent = `${value} profils/page`;
    option.selected = value === state.pageSize;
    pageSize.appendChild(option);
  });
  renderCards(); populateParity(); applyFilters();
}

['search','statusFilter','parityFilter','flipFilter','sortKey','sortDirection'].forEach(id => document.getElementById(id).addEventListener('input', applyFilters));
document.getElementById('closeDialog').addEventListener('click', () => document.getElementById('detailDialog').close());
document.getElementById('detailDialog').addEventListener('click', event => { if (event.target.id === 'detailDialog') event.target.close(); });
document.getElementById('rows').addEventListener('click', event => {
  const row = event.target.closest('tr[data-id]');
  if (row) showDetail(Number(row.dataset.id));
});
document.getElementById('previousPage').addEventListener('click', () => { state.page -= 1; renderRows(); });
document.getElementById('nextPage').addEventListener('click', () => { state.page += 1; renderRows(); });
document.getElementById('pageSize').addEventListener('change', event => {
  state.pageSize = Number(event.target.value); state.page = 1; renderRows();
});
init().catch(error => { document.getElementById('sourceLine').textContent = `Erreur: ${error.message}`; console.error(error); });
</script>
</body>
</html>'''

HTML_PAGE = HTML_PAGE.replace(
    "__WEB_DEFAULT_PAGE_SIZE__", str(settings.WEB_DEFAULT_PAGE_SIZE)
).replace(
    "__WEB_PAGE_SIZE_OPTIONS__", json.dumps(list(settings.WEB_PAGE_SIZE_OPTIONS))
)


def load_payload(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The results JSON root must be an object")

    if isinstance(raw.get("profiles"), list):
        payload = raw
    else:
        raise ValueError(
            f"Unsupported JSON format. Run {settings.AUDIT_SCRIPT_FILENAME} to create {settings.AUDIT_PROFILES_FILENAME}."
        )

    metadata = dict(payload.get("metadata") or {})
    if metadata.get("schema_version") != settings.WEB_SCHEMA_VERSION:
        raise ValueError(
            f"Wrong or stale result schema. Expected {settings.WEB_SCHEMA_VERSION}; "
            f"run {settings.AUDIT_SCRIPT_FILENAME}."
        )
    for profile in payload["profiles"]:
        solution = profile.get("solution", {})
        if not solution.get("profile") or not solution.get("formal_profile"):
            raise ValueError(
                f"Profile {profile.get('profile_id')} has no complete point-and-curve profile"
            )
        if "A" in solution or "B" in solution:
            raise ValueError(
                f"Profile {profile.get('profile_id')} uses the legacy A/B export"
            )
    metadata["source_file"] = str(path.resolve())
    payload["metadata"] = metadata
    return payload


class ResultsHandler(BaseHTTPRequestHandler):
    results_path: Path = DEFAULT_RESULTS_FILE

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/data":
            self._send_file(self.results_path, "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send(b'{"status":"ok"}', "application/json; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Source-File", path.name)
        self.end_headers()
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(settings.WEB_STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local searchable view of solver result JSON.")
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Profile JSON file; defaults to the final-survivor export.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.json.exists():
        raise SystemExit(
            f"Results file not found: {args.json}. Run {settings.AUDIT_SCRIPT_FILENAME} first."
        )
    ResultsHandler.results_path = args.json.resolve()
    server = ThreadingHTTPServer((args.host, args.port), ResultsHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {args.json} at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"use strict";
/* anti-clickjacking: la app nunca debe mostrarse dentro de un marco ajeno */
if(window.top!==window.self){document.documentElement.innerHTML="";throw new Error("frame")}
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const API=()=>window.pywebview&&window.pywebview.api;
let CFG={tono_max:15,pasos:4,max_muestras:25},PEOPLE=[];

/* ---------- comunicación con el backend (capa única) ---------- */
function toast(m,k="ok"){const t=document.createElement("div");t.className="toast "+k;t.textContent=m;$("#toasts").append(t);setTimeout(()=>t.remove(),4200)}
async function rpc(quiet,fn,...a){
  if(!API()){$("#banner").hidden=false;$("#banner").textContent="Sin conexión con el backend. Abre la app con: python -m frontend.app";return{ok:false,error:"Sin conexión con el backend",code:"offline"}}
  try{const r=await API()[fn](...a);if(!r.ok&&r.code==="auth"){lockUI(r.error);return r}if(!r.ok&&!quiet)toast(r.error,r.code==="mic_unavailable"?"err":"warn");return r}
  catch(e){toast("Sin conexión con el backend","err");return{ok:false,error:String(e),code:"offline"}}
}
const call=(f,...a)=>rpc(false,f,...a),callq=(f,...a)=>rpc(true,f,...a);

function modal({title,body="",input=null,ok="Aceptar",cancel="Cancelar",danger=false}){
  return new Promise(res=>{const o=document.createElement("div");o.className="overlay";
    o.innerHTML=`<div class="card modal"><h3>${esc(title)}</h3><div>${body}</div>${input!==null?`<input id="mi" value="${esc(input)}">`:""}<div class="row end">${cancel?`<button class="btn ghost" data-x="0">${cancel}</button>`:""}<button class="btn ${danger?"danger":""}" data-x="1">${ok}</button></div></div>`;
    document.body.append(o);const mi=$("#mi",o);if(mi)mi.focus();
    o.onclick=e=>{const x=e.target.dataset.x;if(x===undefined)return;o.remove();res(x==="1"?(mi?mi.value.trim()||true:true):false)}});
}

/* ---------- gato mascota ---------- */
const catHTML=()=>`<svg class="cat" viewBox="0 0 200 190">
<path d="M45 190Q45 128 100 128Q155 128 155 190Z" fill="#0f1a33" stroke="#22e3ff" stroke-width="3"/>
<g class="ear l"><path d="M44 74L50 18L90 50Z" fill="#0f1a33" stroke="#22e3ff" stroke-width="3"/><path d="M53 58L56 34L72 48Z" fill="#a86bff55"/></g>
<g class="ear r"><path d="M156 74L150 18L110 50Z" fill="#0f1a33" stroke="#22e3ff" stroke-width="3"/><path d="M147 58L144 34L128 48Z" fill="#a86bff55"/></g>
<ellipse cx="100" cy="94" rx="58" ry="48" fill="#0f1a33" class="glow"/>
<g class="eye"><ellipse cx="76" cy="90" rx="13" ry="15" fill="#e8fbff"/><circle class="pupil" cx="76" cy="90" r="7" fill="#0b1224"/><rect class="lid" x="60" y="72" width="32" height="34" fill="#0f1a33"/></g>
<g class="eye"><ellipse cx="124" cy="90" rx="13" ry="15" fill="#e8fbff"/><circle class="pupil" cx="124" cy="90" r="7" fill="#0b1224"/><rect class="lid" x="108" y="72" width="32" height="34" fill="#0f1a33"/></g>
<path d="M95 108L105 108L100 114Z" fill="#a86bff"/><path d="M100 114Q94 122 87 118M100 114Q106 122 113 118" stroke="#22e3ff" stroke-width="2" fill="none"/>
<path d="M60 108L30 102M60 114L30 118M140 108L170 102M140 114L170 118" stroke="#22e3ff88" stroke-width="2"/></svg>`;
addEventListener("mousemove",e=>$$(".cat .eye").forEach(eye=>{const r=eye.getBoundingClientRect(),dx=e.clientX-(r.left+r.width/2),dy=e.clientY-(r.top+r.height/2),d=Math.hypot(dx,dy)||1,k=Math.min(5,d/40);eye.querySelector(".pupil").setAttribute("transform",`translate(${dx/d*k} ${dy/d*k})`)}));
const setCat=on=>$$(".cat").forEach(c=>c.classList.toggle("rec",on));

/* ---------- panel de grabación: onda + escala dBFS reales ---------- */
const recHTML=()=>`<div class="rec"><div class="rec-top"><span class="dot"></span><b class="rs">En espera</b><span class="rt">0.0 s</span></div>
<canvas class="wave" width="640" height="110"></canvas>
<div class="meter"><div class="needle"></div></div><div class="ticks">${[-60,-48,-36,-24,-12,0].map(t=>`<span>${t}</span>`).join("")}</div>
<div class="dbread"><span class="dbv">-- dBFS</span><span class="lvl">Bajo &lt; −40 · Normal · Alto &gt; −12</span></div></div>`;
const lvlOf=d=>d==null?"—":d<-40?"Bajo":d>-12?"Alto":"Normal";
function drawWave(c,w){const g=c.getContext("2d"),W=c.width,H=c.height;g.clearRect(0,0,W,H);g.strokeStyle="#22e3ff";g.lineWidth=2;g.shadowColor="#22e3ff";g.shadowBlur=8;g.beginPath();
  (w.length?w:[0,0]).forEach((v,i,a)=>{const x=i/(a.length-1)*W,y=H/2-Math.max(-1,Math.min(1,v*4))*H/2;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
function drawLine(c,arr,min,max,col){const g=c.getContext("2d"),W=c.width,H=c.height;g.clearRect(0,0,W,H);g.strokeStyle="#1c2a48";g.lineWidth=1;
  for(let i=1;i<4;i++){g.beginPath();g.moveTo(0,H*i/4);g.lineTo(W,H*i/4);g.stroke()}
  g.strokeStyle=col;g.lineWidth=2;g.beginPath();let pen=false;
  arr.forEach((v,i)=>{if(v==null){pen=false;return}const x=i/149*W,y=H-(Math.max(min,Math.min(max,v))-min)/(max-min)*H;pen?g.lineTo(x,y):g.moveTo(x,y);pen=true});g.stroke()}

let rec={on:false,root:null,timer:null,max:15,hist:{db:[],f0:[]},onTick:null,onLimit:null,fired:false};
function liveStart(root,o){rec={on:true,root,max:o.max||15,hist:{db:[],f0:[]},onTick:o.onTick,onLimit:o.onLimit,fired:false,timer:setInterval(tick,100)};
  $(".rec",root).classList.add("on");$(".rs",root).textContent="● Grabando";setCat(true)}
function liveStop(){if(!rec.on)return;clearInterval(rec.timer);rec.on=false;const r=rec.root;$(".rec",r).classList.remove("on");$(".rs",r).textContent="En espera";setCat(false)}
async function tick(){
  if(!rec.on||!API())return;const r=await API().get_live();if(!r.ok||!rec.on)return;const l=r.data,root=rec.root;
  drawWave($(".wave",root),l.wave);$(".rt",root).textContent=`${l.elapsed.toFixed(1)} / ${rec.max} s`;
  const d=l.db==null?-60:Math.max(-60,Math.min(0,l.db));$(".needle",root).style.left=`calc(${(d+60)/60*100}% - 2px)`;
  $(".dbv",root).textContent=l.db==null?"-- dBFS":`${l.db.toFixed(1)} dBFS`;
  rec.hist.db.push(l.db);rec.hist.f0.push(l.f0);Object.values(rec.hist).forEach(a=>a.length>150&&a.shift());
  if(rec.onTick)rec.onTick(l,rec.hist);
  if(l.elapsed>=rec.max&&!rec.fired){rec.fired=true;rec.onLimit&&rec.onLimit()}
}
const en=(id,on)=>{$(id).disabled=!on};

/* ---------- navegación ---------- */
const init={};
async function show(v){
  if(rec.on){liveStop();await callq("record_cancel")}
  $$(".view").forEach(x=>x.hidden=x.id!=="v-"+v);$$(".nav button").forEach(b=>b.classList.toggle("on",b.dataset.v===v));
  if(init[v])init[v]();
}
async function doExit(){if(await modal({title:"Salir",body:"Se liberará el micrófono y se cerrará la aplicación.",ok:"Salir",danger:true}))call("exit")}

/* ---------- registro / re-entrenamiento (asistente de 4 pasos) ---------- */
let W=null;
const MODO_TXT={registro:"Registrar",reentrenar:"Re-entrenar",ampliar:"Añadir muestras"};
function renderWizard(i){W=i;$("#reg-setup").hidden=true;$("#reg-run").hidden=false;
  $("#reg-title").textContent=`${MODO_TXT[i.modo]||"Registrar"}: ${i.nombre}`+(i.lleno?"":` · muestra ${i.paso}`);
  const dots=Math.max(i.min,Math.min(i.max,i.guardadas+1));
  $("#reg-steps").innerHTML=Array.from({length:dots},(_,k)=>`<i class="${k<i.guardadas?"done":k===i.guardadas?"now":""}"></i>`).join("");
  $("#reg-text").textContent=i.lleno?"Se alcanzó el máximo de muestras. Pulsa «Completar registro».":i.texto;
  $("#reg-count").textContent=`${i.guardadas} nueva(s) · mínimo ${i.min}, máximo ${i.max}`+(i.modo==="ampliar"?` · total de la persona: ${i.existentes+i.guardadas}/${CFG.max_muestras}`:"")+(i.completo&&!i.lleno?" · puedes completar ya o grabar más (opcional)":"");
  $("#reg-finish").hidden=!i.completo;en("#reg-start",!i.lleno);en("#reg-stop",false);$("#reg-result").innerHTML=""}
async function beginWizard(modo,nombre){const r=await call("wizard_start",nombre,modo);if(r.ok)renderWizard(r.data)}
init.registro=()=>{if(!W){$("#reg-setup").hidden=false;$("#reg-run").hidden=true;$("#reg-name").value="";$("#reg-count").textContent=""}};
$("#reg-begin").onclick=()=>beginWizard("registro",$("#reg-name").value);
$("#reg-name").onkeydown=e=>{if(e.key==="Enter")$("#reg-begin").click()};
$("#reg-start").onclick=async()=>{const r=await call("record_start");if(!r.ok)return;$("#reg-result").innerHTML="";en("#reg-start",false);en("#reg-stop",true);liveStart($("#reg-rec"),{max:W.duracion,onLimit:regStop})};
async function regStop(){if(!rec.on)return;liveStop();en("#reg-stop",false);const r=await call("record_stop");en("#reg-start",true);if(!r.ok)return;
  $("#reg-result").innerHTML=r.data.valida
    ?`<p class="status ok">✔ Muestra válida (${r.data.segundos} s)</p><div class="row"><button class="btn ok" id="k-y">Conservar</button><button class="btn ghost" id="k-n">Repetir</button></div>`
    :`<p class="status err">✖ Muestra rechazada: ${esc(r.data.motivo)}</p>`;
  const y=$("#k-y");if(y){en("#reg-start",false);y.onclick=async()=>{const k=await call("wizard_keep");if(k.ok)renderWizard(k.data)};$("#k-n").onclick=()=>{$("#reg-result").innerHTML="";en("#reg-start",!W.lleno)}}}
$("#reg-stop").onclick=regStop;
$("#reg-finish").onclick=async()=>{en("#reg-finish",false);$("#reg-result").innerHTML=`<p class="status">Procesando y guardando (extrayendo embeddings)…</p>`;
  const r=await call("wizard_finish");en("#reg-finish",true);if(!r.ok){$("#reg-result").innerHTML="";return}
  toast(`${r.data.guardadas} muestra(s) guardada(s) para ${r.data.nombre}`);W=null;init.registro()};
$("#reg-cancel").onclick=async()=>{if(rec.on){liveStop()}await callq("wizard_cancel");W=null;toast("Operación cancelada","warn");init.registro()};

/* ---------- identificar ---------- */
let idDur=10;
const idStatus=(m,k="")=>{const s=$("#id-status");s.textContent=m;s.className="status "+k};
init.identificar=async()=>{$("#id-result").innerHTML=`<h3>Resultado de identificación</h3><p class="muted">Aún no hay resultado.</p>`;idStatus("");
  const r=await call("identify_info");en("#id-start",r.ok);en("#id-stop",false);
  if(!r.ok){$("#id-text").textContent="—";idStatus(r.error,"err");return}
  $("#id-text").textContent=r.data.texto;idDur=r.data.duracion;
  if(r.data.desactualizados.length)idStatus(`Sin comparar (desactualizados): ${r.data.desactualizados.join(", ")}. Re-entrénalos.`)};
$("#id-start").onclick=async()=>{const r=await call("record_start");if(!r.ok)return idStatus(r.error,"err");en("#id-start",false);en("#id-stop",true);idStatus("Grabando…");liveStart($("#id-rec"),{max:idDur,onLimit:idStop})};
async function idStop(){if(!rec.on)return;liveStop();en("#id-stop",false);idStatus("Procesando…");
  const r=await call("record_stop");if(!r.ok||!r.data.valida){en("#id-start",true);idStatus(r.ok?`No se pudo analizar: ${r.data.motivo}`:r.error,"err");
    if(r.ok)$("#id-result").innerHTML=`<h3>Resultado de identificación</h3><p class="fail">No se reconoce ninguna voz registrada</p>`;return}
  const q=await call("identify_run");en("#id-start",true);if(!q.ok)return idStatus(q.error,"err");idStatus("");const d=q.data;
  $("#id-result").innerHTML=d.identificado
    ?`<h3>Resultado de identificación</h3><p>Persona identificada:</p><div class="big">${esc(d.nombre)}</div><p>Confianza: <b class="ok-c">${(d.similitud*100).toFixed(1)}%</b> · distancia ${d.distancia.toFixed(3)}</p>${d.vivacidad_baja?`<p class="warn-c">⚠ Vivacidad baja: heurística informativa, verifica en persona.</p>`:""}`
    :`<h3>Resultado de identificación</h3><p class="fail"><b>No se reconoce ninguna voz registrada</b></p><p class="muted">Mejor candidato (no aceptado): ${esc(d.nombre)} · similitud ${(d.similitud*100).toFixed(1)}% (umbral ${(d.umbral_sim*100).toFixed(1)}%) · distancia ${d.distancia==null?"—":d.distancia.toFixed(3)}</p>`}
$("#id-stop").onclick=idStop;

/* ---------- personas: eliminar / gestionar ---------- */
const badge=p=>p.reentrenar?`<span class="badge w">Desactualizada</span>`:p.compatibles<p.muestras?`<span class="badge w">Parcial</span>`:`<span class="badge">Vigente</span>`;
async function loadPeople(){const r=await call("list_speakers");PEOPLE=r.ok?r.data:[];return PEOPLE}
async function deletePerson(p,after){
  if(!await modal({title:"Eliminar persona",body:`¿Eliminar a <b>${esc(p.nombre)}</b> (${esc(p.muestras)} muestras de voz)? Se borrarán sus datos biométricos y esta acción no se puede deshacer.`,ok:"Eliminar",danger:true}))return toast("Operación cancelada","warn");
  const r=await call("delete_speaker",p.nombre);if(r.ok){toast(`${p.nombre} eliminada (${r.data} archivos borrados)`);after()}}
let SEL=null;
init.eliminar=async()=>{SEL=null;en("#del-go",false);const p=await loadPeople();
  $("#del-list").innerHTML=p.length?p.map(x=>`<div class="item" data-n="${esc(x.nombre)}"><div class="grow"><b>${esc(x.nombre)}</b><div class="muted">${esc(x.muestras)} muestras · ${esc(x.fecha)}</div></div>${badge(x)}</div>`).join(""):`<p class="muted">No hay personas registradas.</p>`};
$("#del-list").onclick=e=>{const it=e.target.closest(".item");if(!it)return;SEL=PEOPLE.find(p=>p.nombre===it.dataset.n);$$("#del-list .item").forEach(x=>x.classList.toggle("sel",x===it));en("#del-go",true)};
$("#del-go").onclick=()=>SEL&&deletePerson(SEL,init.eliminar);
init.gestionar=async()=>{const p=await loadPeople();
  $("#mg-body").innerHTML=p.length?p.map(x=>`<tr><td>${esc(x.nombre)}</td><td>${esc(x.muestras)} / ${esc(CFG.max_muestras)}</td><td>${badge(x)}</td><td>${esc(x.fecha)}</td><td class="acts" data-n="${esc(x.nombre)}"><button data-a="info">Info</button><button data-a="ren">Renombrar</button><button data-a="re">Re-entrenar</button><button data-a="add"${x.muestras>=CFG.max_muestras?" disabled":""}>Añadir muestras</button><button data-a="del" class="danger">Eliminar</button></td></tr>`).join(""):`<tr><td colspan="5" class="muted">No hay personas registradas.</td></tr>`};
$("#mg-body").onclick=async e=>{const a=e.target.dataset.a,td=e.target.closest(".acts");if(!a||!td)return;const n=td.dataset.n,p=PEOPLE.find(x=>x.nombre===n);
  if(!p)return;
  if(a==="info")modal({title:n,body:`<p>Registro: ${esc(p.fecha)}<br>Muestras: ${esc(p.muestras)} de máx. ${esc(CFG.max_muestras)} (${esc(p.compatibles)} compatibles con el motor actual)<br>Estado: ${badge(p)}</p>`,ok:"Cerrar",cancel:""});
  if(a==="ren"){const nn=await modal({title:"Renombrar",input:n,ok:"Guardar"});if(nn&&nn!==true&&nn!==n){const r=await call("rename_speaker",n,nn);if(r.ok){toast("Persona renombrada");init.gestionar()}}}
  if(a==="re"&&await modal({title:"Re-entrenar",body:`Se grabarán de ${esc(CFG.pasos)} a ${esc(CFG.max_muestras)} muestras nuevas para <b>${esc(n)}</b>. Las actuales solo se reemplazan al completar.`,ok:"Continuar"})){W=null;await show("registro");beginWizard("reentrenar",n)}
  if(a==="add"){W=null;await show("registro");beginWizard("ampliar",n)}
  if(a==="del")deletePerson(p,init.gestionar)};

/* ---------- análisis de voz ---------- */
function anTick(l,h){$("#an-f0").textContent=l.f0?`${l.f0.toFixed(0)} Hz`:"— Hz";$("#an-tono").textContent=l.f0?l.tono:"Sin datos";
  $("#an-vol").textContent=l.db==null?"— dBFS":`${l.db.toFixed(1)} dBFS`;$("#an-lvl").textContent=lvlOf(l.db);
  drawLine($("#an-c1"),h.f0,50,400,"#a86bff");drawLine($("#an-c2"),h.db,-60,0,"#22e3ff")}
init.analisis=()=>{$("#an-sum").hidden=true};
$("#an-start").onclick=async()=>{const r=await call("analysis_start");if(!r.ok)return;$("#an-sum").hidden=true;en("#an-start",false);en("#an-stop",true);liveStart($("#an-rec"),{max:CFG.tono_max,onTick:anTick,onLimit:anStop})};
async function anStop(){if(!rec.on)return;liveStop();en("#an-stop",false);en("#an-start",true);const r=await call("analysis_stop");if(!r.ok)return;const s=r.data;
  $("#an-sum").hidden=false;$("#an-sum").innerHTML=s.f0>0
    ?`<h3>Resumen (${s.segundos} s)</h3><p>F0 promedio: <b>${s.f0.toFixed(1)} Hz</b> · Clasificación: <b>${s.tono}</b> · Volumen promedio: <b>${s.db==null?"—":s.db.toFixed(1)+" dBFS"}</b> (RMS ${s.rms.toFixed(4)})</p>${s.segundos<CFG.tono_min?`<p class="warn-c">Grabación corta (&lt; ${CFG.tono_min} s): el promedio de F0 puede ser poco confiable.</p>`:""}`
    :`<p class="warn-c">No se detectó suficiente señal de voz para analizar el tono. Revisa el micrófono o habla más fuerte y cerca.</p>`}
$("#an-stop").onclick=anStop;

/* ---------- pruebas automáticas ---------- */
const TESTS=[["Micrófono","Captura 2 s y comprueba que hay señal"],["Base de datos","Escribe, relee y limpia un registro de prueba"],["Modelo de voz","Extrae embeddings y compara consigo mismo"]];
const IC={pend:["○","pend"],run:["◐","run"],ok:["✔","ok-c"],warn:["⚠","warn-c"],fail:["✖","fail"]};
function renderTests(st){$("#pt-list").innerHTML=TESTS.map((t,i)=>{const s=st[i]||{s:"pend"},[c,k]=IC[s.s];return`<div class="item"><span class="t-ic ${k}">${c}</span><div class="grow"><b>${t[0]}</b><div class="muted">${esc(s.d||t[1])}</div></div><span class="muted">${s.t!=null?s.t+" s":""}</span></div>`}).join("")}
init.pruebas=()=>{renderTests([]);$("#pt-bar").style.width="0";$("#pt-sum").textContent=""};
$("#pt-run").onclick=async()=>{en("#pt-run",false);const st=[];let bad=0,warn=0;const t0=performance.now();$("#pt-sum").textContent="Ejecutando…";
  for(let i=0;i<TESTS.length;i++){st[i]={s:"run",d:"Ejecutando…"};renderTests(st);
    const r=await callq("run_test",i+1);
    if(!r.ok){bad++;st[i]={s:"fail",d:r.error}}else{if(r.data.estado==="warn")warn++;st[i]={s:r.data.estado,d:r.data.detalle,t:r.data.segundos}}
    renderTests(st);$("#pt-bar").style.width=`${(i+1)/TESTS.length*100}%`}
  $("#pt-sum").innerHTML=`${bad?`<span class="fail">✖ ${bad} prueba(s) fallida(s)</span>`:warn?`<span class="warn-c">⚠ Completado con advertencias</span>`:`<span class="ok-c">✔ Todas las pruebas pasaron</span>`} · ${((performance.now()-t0)/1000).toFixed(1)} s`;en("#pt-run",true)};

/* ---------- arranque + PIN ---------- */
$$("[data-rec]").forEach(e=>e.innerHTML=recHTML());$$("[data-cat]").forEach(e=>e.innerHTML=catHTML());
$$(".nav button").forEach(b=>b.onclick=()=>b.dataset.v==="salir"?doExit():b.dataset.v==="bloquear"?doLock():show(b.dataset.v));
function lockUI(msg){if(rec.on)liveStop();W=null;$$(".overlay:not(#lock)").forEach(o=>o.remove());$("#lock").hidden=false;$("#pin").value="";
  $("#pinmsg").textContent=msg||"Introduce tu PIN";$("#pinmsg").className="status"+(msg?" warn-c":"");$("#pin").focus()}
async function doLock(){await callq("logout");lockUI("Sesión bloqueada")}
$("#pinform").onsubmit=async e=>{e.preventDefault();const r=await callq("login",$("#pin").value);$("#pin").value="";
  if(r.ok){$("#lock").hidden=true;W=null;show("registro")}else{$("#pinmsg").textContent=r.error;$("#pinmsg").className="status err"}};
async function boot(){
  const c=await callq("get_config");if(c.ok)CFG=c.data;
  const a=await callq("auth_status");if(!a.ok)return;
  if(!a.data.configured){$("#lock").hidden=false;$("#pinmsg").textContent="No hay PIN configurado. Ejecuta «python setup_seguridad.py» y reinicia la app.";$("#pin").hidden=true;$("#pinform").querySelector(".row").hidden=true;return}
  if(a.data.authenticated){show("registro")}else{$("#lock").hidden=false;$("#pin").focus()}
}
addEventListener("pywebviewready",boot);
setTimeout(()=>{if(!API()){$("#banner").hidden=false;$("#banner").textContent="Sin conexión con el backend. Abre la app con: python -m frontend.app"}},1500);

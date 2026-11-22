(function (global) {
'use strict';
const LAYERS = [
 {name:'Block 1',input:'3 × 32 × 32',output:'32 × 16 × 16',params:10272,macs:10321920,m:3,n:32,width:32},
 {name:'Block 2',input:'32 × 16 × 16',output:'64 × 8 × 8',params:55680,macs:14155776,m:32,n:64,width:16},
 {name:'Block 3',input:'64 × 8 × 8',output:'128 × 4 × 4',params:221952,macs:14155776,m:64,n:128,width:8},
 {name:'Block 4',input:'128 × 4 × 4',output:'256 × 2 × 2',params:886272,macs:14155776,m:128,n:256,width:4},
 {name:'Classifier',input:'256 × 2 × 2',output:'10 logits',params:264970,macs:264704}
];
function quantize(value, mode='export') {
 if(!Number.isFinite(value)||Math.abs(value)>1000000) throw Error('Enter a finite value between −1,000,000 and 1,000,000.');
 if(!['export','activation'].includes(mode)) throw Error('Unknown conversion rule.');
 const scaled=value*16, lower=Math.floor(scaled), fraction=scaled-lower;
 const rounded=mode==='activation'?lower:fraction<.5?lower:fraction>.5?lower+1:lower%2===0?lower:lower+1;
 const overflow=rounded< -32768||rounded>32767;
 const raw=mode==='export'?Math.max(-32768,Math.min(32767,rounded)):((rounded+32768)%65536+65536)%65536-32768;
 const bits=((raw+65536)%65536).toString(2).padStart(16,'0');
 return {raw,value:raw/16,error:raw/16-value,overflow,bits,hex:'0x'+((raw+65536)%65536).toString(16).toUpperCase().padStart(4,'0')};
}
function tileStats(width,tile,m,n) {
 if(![4,8,16,32].includes(width)||![2,4,8].includes(tile)||!Number.isInteger(m)||m<1||!Number.isInteger(n)||n<1) throw Error('Unsupported tile configuration.');
 const across=Math.ceil(width/tile);let reads=0;
 for(let r=0;r<across;r++)for(let c=0;c<across;c++){
  const h=Math.max(0,Math.min(width,r*tile+tile+1)-Math.max(0,r*tile-1));
  const w=Math.max(0,Math.min(width,c*tile+tile+1)-Math.max(0,c*tile-1));
  reads+=h*w*m;
 }
 return {across,tiles:across*across,buffer:(tile+2)**2*m,reads,inputBytes:reads*2,weightBytes:across**2*m*n*9*2};
}
function finite(value){return typeof value==='number'&&Number.isFinite(value);}
function validateData(data){
 const m=data.metrics;
 if(!m||m.samples!==10000||!Array.isArray(m.per_class)||m.per_class.length!==10||!Array.isArray(m.confusion_matrix)||m.confusion_matrix.length!==10) throw Error('The recorded classification data is incomplete.');
 let total=0,correct=0;
 m.confusion_matrix.forEach((row,r)=>{
  if(!Array.isArray(row)||row.length!==10||row.some(v=>!Number.isInteger(v)||v<0))throw Error('Invalid confusion matrix.');
  total+=row.reduce((a,b)=>a+b,0);correct+=row[r];
  if(m.per_class[r].support!==row.reduce((a,b)=>a+b,0))throw Error('Class support does not match the matrix.');
  for(const key of ['precision','recall','f1'])if(!finite(m.per_class[r][key])||m.per_class[r][key]<0||m.per_class[r][key]>1)throw Error('Invalid class metric.');
 });
 if(total!==m.samples||Math.abs(correct/total-m.accuracy)>1e-10)throw Error('Classification totals do not agree.');
 if(!Array.isArray(m.selective_predictions)||m.selective_predictions.length!==7||m.selective_predictions.some(x=>!finite(x.threshold)||x.threshold<0||x.threshold>1||x.accepted+x.rejected!==total||!finite(x.coverage)||!finite(x.accuracy)||!finite(x.macro_recall_counting_rejections_as_misses)))throw Error('Invalid confidence data.');
 if(!data.training||!Array.isArray(data.training.history)||!data.training.history.length||data.training.history.some(x=>!['fp32','qat'].includes(x.phase)||!finite(x.validation_accuracy)||!finite(x.train_accuracy)||!finite(x.loss)))throw Error('Training history is unavailable.');
 if(!data.ranges||!data.ranges.layers||!Object.keys(data.ranges.layers).length||Object.values(data.ranges.layers).some(x=>!finite(x.minimum)||!finite(x.maximum)||x.minimum>x.maximum))throw Error('Range audit is unavailable.');
 if(!data.verification||!data.verification.golden_vector||!Array.isArray(data.verification.golden_vector.logits)||data.verification.golden_vector.logits.length!==10||data.verification.golden_vector.logits.some(x=>!finite(x)))throw Error('Verification results are unavailable.');
 return data;
}
const API={LAYERS,quantize,tileStats,validateData};
if(typeof module!=='undefined'&&module.exports)module.exports=API;
if(typeof document==='undefined')return;
global.NeuroForge=API;
const $=id=>document.getElementById(id),$$=selector=>Array.from(document.querySelectorAll(selector));
const number=n=>Number(n).toLocaleString('en-US'),percent=(v,d=2)=>(v*100).toFixed(d)+'%';
const state={data:null,layer:0,focus:3,cell:[3,3],matrix:'count',request:0};
const SVG='http://www.w3.org/2000/svg';
function el(tag,attrs={},text){const node=document.createElement(tag);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,String(v));if(text!==undefined)node.textContent=text;return node;}
function s(tag,attrs={},text){const node=document.createElementNS(SVG,tag);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,String(v));if(text!==undefined)node.textContent=text;return node;}
function on(id,event,fn){if($(id))$(id).addEventListener(event,fn);}
function set(id,text){if($(id))$(id).textContent=text;}
function applyTheme(preference,save=true){
 const dark=global.matchMedia&&global.matchMedia('(prefers-color-scheme: dark)').matches;
 const theme=preference==='system'?(dark?'dark':'light'):preference;
 document.documentElement.dataset.theme=theme;document.documentElement.dataset.preference=preference;
 if(save)try{global.localStorage.setItem('neuroforge-theme',preference);}catch{}
 set('theme-label',theme==='dark'?'Light':'Dark');
 if($('theme-toggle'))$('theme-toggle').setAttribute('aria-label','Switch to '+(theme==='dark'?'light':'dark')+' mode');
 if($('theme-system')){$('theme-system').textContent=preference==='system'?'System theme active':'Use system theme';$('theme-system').disabled=preference==='system';}
}
function initTheme(){
 applyTheme(document.documentElement.dataset.preference||'system',false);
 on('theme-toggle','click',()=>applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
 on('theme-system','click',()=>applyTheme('system'));
 const media=global.matchMedia?global.matchMedia('(prefers-color-scheme: dark)'):null;
 if(media&&media.addEventListener)media.addEventListener('change',()=>{if(document.documentElement.dataset.preference==='system')applyTheme('system',false);});
}
function renderLayer(){
 const row=LAYERS[state.layer];
 $$('[data-layer]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.layer)===state.layer)));
 set('layer-name',row.name);set('layer-input',row.input);set('layer-output',row.output);
 set('layer-params',number(row.params));set('layer-macs',number(row.macs));
 set('layer-description',state.layer===4?'The pooled 256 × 2 × 2 feature map flattens into 1,024 values. A 256-unit fully connected layer and ReLU feed the final ten class scores.':'Two 3×3 convolutions, each followed by BatchNorm and ReLU. A 2×2 max pool halves the spatial dimensions.');
}
function renderTile(){
 if(!$('tile-viz'))return;
 const layer=LAYERS[Number($('tile-layer').value)];
 Array.from($('tile-size').options).forEach(option=>{option.disabled=Number(option.value)>layer.width;});
 if(Number($('tile-size').value)>layer.width)$('tile-size').value=String(layer.width);
 const tile=Number($('tile-size').value),stats=tileStats(layer.width,tile,layer.m,layer.n);
 $('tile-position').max=stats.tiles-1;
 const pos=Math.min(Number($('tile-position').value),stats.tiles-1);$('tile-position').value=pos;
 const tr=Math.floor(pos/stats.across),tc=pos%stats.across;
 const svg=$('tile-viz');svg.replaceChildren();
 const size=Math.min(218,240*layer.width/(layer.width+2)),cell=size/layer.width,x=(390-size)/2,y=(260-size)/2;
 svg.append(s('rect',{x,y,width:size,height:size,fill:'var(--paper)',stroke:'var(--line)'}));
 for(let r=0;r<layer.width;r++)for(let c=0;c<layer.width;c++){
  const inCore=r>=tr*tile&&r<(tr+1)*tile&&c>=tc*tile&&c<(tc+1)*tile;
  const inHalo=r>=tr*tile-1&&r<(tr+1)*tile+1&&c>=tc*tile-1&&c<(tc+1)*tile+1;
  svg.append(s('rect',{x:x+c*cell+.4,y:y+r*cell+.4,width:Math.max(1,cell-.8),height:Math.max(1,cell-.8),fill:inCore?'var(--accent)':inHalo?'var(--accent-soft)':'var(--soft)'}));
 }
 const cx=x+tc*tile*cell,cy=y+tr*tile*cell;
 svg.append(s('rect',{x:cx-cell,y:cy-cell,width:(tile+2)*cell,height:(tile+2)*cell,fill:'none',stroke:'var(--muted)','stroke-dasharray':'4 3'}));
 svg.append(s('text',{x:70,y:16,'text-anchor':'end',fill:'var(--muted)','font-size':10,'font-family':'var(--mono)'},'0'));
 svg.append(s('text',{x:70,y:236,'text-anchor':'end',fill:'var(--muted)','font-size':10,'font-family':'var(--mono)'},String(layer.width-1)));
 set('tile-coordinate','row '+tr+' · col '+tc);set('tile-position-label',(pos+1)+' / '+stats.tiles);
 set('tile-count',number(stats.tiles));set('tile-buffer',number((tile+2)**2)+' × '+layer.m);set('tile-reads',number(stats.reads));
 svg.setAttribute('aria-label',layer.width+' by '+layer.width+' feature map, '+tile+' by '+tile+' output tile at row '+tr+' column '+tc+', with a one-pixel halo');
}
function renderQuant(){
 if(!$('quant-bits'))return;
 try{
  if($('quant-value').value.trim()==='')throw Error('Enter a value to inspect its representation.');
  const q=quantize(Number($('quant-value').value),$('quant-mode').value);
  const bits=Array.from(q.bits,(bit,i)=>el('span',{class:'bit '+(i===0?'sign':i>=12?'fraction':''),'aria-hidden':'true'},bit));
  $('quant-bits').replaceChildren(...bits);$('quant-bits').setAttribute('aria-label',q.bits+'; '+q.hex);
  set('quant-output',q.value.toFixed(4));set('quant-hex',q.hex);set('quant-raw',number(q.raw));
  set('quant-error',(q.error>=0?'+':'')+q.error.toFixed(6));
  set('quant-status',q.overflow?($('quant-mode').value==='export'?'Outside range: clamped to the signed 16-bit limit.':'Outside range: wrapped using the default hardware overflow rule.'):'Error: '+(q.error>=0?'+':'')+q.error.toFixed(6)+' · represented in steps of 0.0625');
  $('quant-status').classList.toggle('warn',q.overflow);
 }catch(error){
  $('quant-bits').replaceChildren();set('quant-output','—');set('quant-hex','—');set('quant-raw','—');set('quant-error','—');set('quant-status',error.message);$('quant-status').classList.add('warn');
 }
}
function updateClassFocus(){
 if($('class-focus'))$('class-focus').value=state.focus;
 $$('[data-class-index]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.classIndex)===state.focus)));
}
function renderBars(){
 if(!$('class-bars')||!state.data)return;
 const metric=$('class-metric').value,order=$('class-sort').value;
 const rows=[...state.data.metrics.per_class];
 if(order!=='class')rows.sort((a,b)=>(order==='lowest'?a[metric]-b[metric]:b[metric]-a[metric])||a.index-b.index);
 const nodes=rows.map(row=>{
  const button=el('button',{type:'button',class:'metric-row','data-class-index':row.index,'aria-pressed':String(row.index===state.focus),'aria-label':row.name+', '+metric+' '+percent(row[metric])});
  const track=el('span',{class:'track','aria-hidden':'true'}),fill=el('span',{class:'fill'});
  fill.style.width=(row[metric]*100)+'%';track.append(fill);
  button.append(el('span',{},row.name),track,el('span',{class:'bar-value'},percent(row[metric])));
  button.addEventListener('click',()=>{state.focus=row.index;state.cell=[row.index,row.index];updateClassFocus();renderMatrix();});
  return button;
 });
 $('class-bars').replaceChildren(...nodes);
}
function renderMatrix(){
 if(!$('confusion-matrix')||!state.data)return;
 const m=state.data.metrics,names=m.class_names,short=['air','auto','bird','cat','deer','dog','frog','horse','ship','truck'];
 const table=$('confusion-matrix'),caption=el('caption',{},'Rows: true class. Columns: predicted class.');
 const head=el('thead'),heading=el('tr');heading.append(el('th',{scope:'col'},'T / P'));
 names.forEach((name,i)=>heading.append(el('th',{scope:'col',title:name},short[i])));head.append(heading);
 const body=el('tbody');
 m.confusion_matrix.forEach((row,r)=>{
  const tr=el('tr',{class:r===state.focus?'focus-row':''});tr.append(el('th',{scope:'row',title:names[r]},short[r]));
  row.forEach((value,c)=>{
   const p=value/m.per_class[r].support;
   const button=el('button',{type:'button','data-row':r,'data-col':c,'data-active':String(state.cell[0]===r&&state.cell[1]===c),'aria-label':'True '+names[r]+', predicted '+names[c]+': '+value+' images ('+percent(p)+')'},state.matrix==='count'?String(value):percent(p,1));
   button.style.background=p>.5?'var(--accent)':'color-mix(in srgb, var(--accent) '+Math.round(7+p*55)+'%, var(--paper))';
   button.style.color=p>.5?'var(--on-accent)':'var(--ink)';
   const td=el('td');td.append(button);tr.append(td);
  });body.append(tr);
 });
 table.replaceChildren(caption,head,body);
 const [r,c]=state.cell,v=m.confusion_matrix[r][c];
 const strong=el('strong',{},number(v)),description=document.createTextNode('true '+names[r]+' → predicted '+names[c]+' · '+percent(v/m.per_class[r].support)+' of '+names[r]+' images');
 $('matrix-detail').replaceChildren(strong,description);
 $$('[data-matrix-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.matrixMode===state.matrix)));
}
function initResults(){
 if(!state.data)return;
 const m=state.data.metrics;
 $$('[data-result]').forEach(node=>{const value=m[node.dataset.result];node.textContent=node.dataset.result.includes('auc')?value.toFixed(5):percent(value);});
 $$('[data-summary]').forEach(node=>{node.replaceChildren(document.createTextNode((m[node.dataset.summary]*100).toFixed(2)),el('small',{},'%'));});
 if($('class-focus')){
  $('class-focus').replaceChildren(...m.class_names.map((name,i)=>el('option',{value:i},name)));updateClassFocus();
 }
 if($('confusion-list'))$('confusion-list').replaceChildren(...m.largest_confusions.slice(0,3).map(row=>{
  const li=el('li'),button=el('button',{type:'button'});
  button.append(el('span',{},row.actual+' → '+row.predicted),el('span',{},number(row.count)+' images'));
  button.addEventListener('click',()=>{state.focus=m.class_names.indexOf(row.actual);state.cell=[state.focus,m.class_names.indexOf(row.predicted)];updateClassFocus();renderMatrix();});
  li.append(button);return li;
 }));
 renderBars();renderMatrix();
}
function chartBase(svg,width,height,domain,formatter){
 svg.replaceChildren();const box={left:45,top:18,width:width-65,height:height-60};
 for(let i=0;i<=4;i++){
  const y=box.top+box.height-i*box.height/4,value=domain[0]+(domain[1]-domain[0])*i/4;
  svg.append(s('line',{x1:box.left,x2:box.left+box.width,y1:y,y2:y,class:'grid'}));
  svg.append(s('text',{x:box.left-9,y:y+3,'text-anchor':'end'},formatter(value)));
 }
 svg.append(s('line',{x1:box.left,x2:box.left+box.width,y1:box.top+box.height,y2:box.top+box.height,class:'axis'}));
 return box;
}
function renderTraining(){
 if(!$('training-chart')||!state.data)return;
 const rows=state.data.training.history,metric=$('training-metric').value,svg=$('training-chart');
 $('epoch-position').max=rows.length-1;
 const index=Math.min(Number($('epoch-position').value),rows.length-1);$('epoch-position').value=index;
 const domain=metric==='loss'?[0,Math.max(...rows.map(x=>x.loss))*1.1]:[.4,1];
 const box=chartBase(svg,700,250,domain,v=>metric==='loss'?v.toFixed(1):Math.round(v*100)+'%');
 const x=i=>box.left+i/(Math.max(1,rows.length-1))*box.width,y=v=>box.top+box.height-(v-domain[0])/(domain[1]-domain[0])*box.height;
 const fpCount=rows.filter(row=>row.phase==='fp32').length;
 if(fpCount&&fpCount<rows.length){const split=x(fpCount-.5);svg.append(s('rect',{x:split,y:box.top,width:box.left+box.width-split,height:box.height,class:'annotation-fill'}));svg.append(s('text',{x:split+8,y:box.top+13},'QAT'));}
 for(const i of Array.from(new Set([0,Math.round((rows.length-1)*.25),Math.round((rows.length-1)*.5),Math.round((rows.length-1)*.75),rows.length-1]))){
  svg.append(s('text',{x:x(i),y:box.top+box.height+20,'text-anchor':'middle'},String(i+1)));
 }
 svg.append(s('text',{x:box.left+box.width,y:244,'text-anchor':'end'},'Recorded epoch'));
 const path=rows.map((row,i)=>(i?'L':'M')+x(i).toFixed(2)+' '+y(row[metric]).toFixed(2)).join(' ');
 svg.append(s('path',{d:path,class:'series'}));
 svg.append(s('line',{x1:x(index),x2:x(index),y1:box.top,y2:box.top+box.height,class:'selection'}));
 svg.append(s('circle',{cx:x(index),cy:y(rows[index][metric]),r:5,class:'point'}));
 const row=rows[index];set('epoch-label',row.phase.toUpperCase()+' · epoch '+row.epoch);set('epoch-validation',percent(row.validation_accuracy));set('epoch-training',percent(row.train_accuracy));set('epoch-loss',row.loss.toFixed(4));
 svg.setAttribute('aria-label',metric.replaceAll('_',' ')+', '+row.phase+' epoch '+row.epoch+', value '+(metric==='loss'?row.loss.toFixed(4):percent(row[metric])));
}
function renderConfidence(){
 if(!$('confidence-chart')||!state.data)return;
 const rows=state.data.metrics.selective_predictions,index=Number($('confidence-threshold').value),row=rows[index],svg=$('confidence-chart');
 const box=chartBase(svg,600,260,[.85,1],v=>Math.round(v*100)+'%');
 const x=v=>box.left+(v-.55)/.45*box.width,y=v=>box.top+box.height-(v-.85)/.15*box.height;
 for(const value of [.6,.7,.8,.9,1])svg.append(s('text',{x:x(value),y:box.top+box.height+20,'text-anchor':'middle'},percent(value,0)));
 svg.append(s('text',{x:box.left+box.width,y:253,'text-anchor':'end'},'Images accepted / coverage'));
 svg.append(s('path',{d:rows.map((r,i)=>(i?'L':'M')+x(r.coverage)+' '+y(r.accuracy)).join(' '),class:'series'}));
 rows.forEach((r,i)=>svg.append(s('circle',{cx:x(r.coverage),cy:y(r.accuracy),r:i===index?6:3,class:'point',style:i===index?'fill:var(--warning)':''})));
 set('confidence-threshold-label',percent(row.threshold,0));set('confidence-coverage',percent(row.coverage));set('confidence-accuracy',percent(row.accuracy));set('confidence-recall',percent(row.macro_recall_counting_rejections_as_misses));set('confidence-rejected',number(row.rejected));
 svg.setAttribute('aria-label','At '+percent(row.threshold,0)+' minimum confidence, coverage '+percent(row.coverage)+', accepted accuracy '+percent(row.accuracy)+' and overall recall '+percent(row.macro_recall_counting_rejections_as_misses));
}
function initConfidence(){
 if(!$('threshold-table')||!state.data)return;
 const body=$('threshold-table').querySelector('tbody');
 body.replaceChildren(...state.data.metrics.selective_predictions.map(row=>{const tr=el('tr');for(const value of [row.threshold,row.coverage,row.accuracy,row.macro_recall_counting_rejections_as_misses])tr.append(el('td',{},percent(value)));return tr;}));renderConfidence();
}
function friendlyLayer(name){
 if(name==='input')return 'Input';
 if(name==='classifier.1')return 'FC 1';if(name==='classifier.4')return 'FC 2';
 const match=name.match(/features\.(\d)\.block\.(\d)/);if(!match)return name;
 return 'B'+(Number(match[1])+1)+' · '+({0:'conv 1',1:'BN 1',3:'conv 2',4:'BN 2'}[match[2]]||match[2]);
}
function renderRanges(){
 if(!$('range-rows')||!state.data)return;
 const bounds=$('range-view').value==='bounds';
 const rows=Object.entries(state.data.ranges.layers).filter(([,row])=>!bounds||finite(row.maximum_absolute_partial_sum_bound));
 const max=Math.max(...rows.flatMap(([,row])=>bounds?[row.maximum_absolute_partial_sum_bound]:[Math.abs(row.minimum),Math.abs(row.maximum)]),1);
 $('range-rows').replaceChildren(...rows.map(([name,row])=>{
  const item=el('div',{class:'range-row'}),track=el('span',{class:'range-track','aria-hidden':'true'}),mark=el('span',{class:'range-mark'});
  const low=bounds?0:row.minimum,high=bounds?row.maximum_absolute_partial_sum_bound:row.maximum;
  mark.style.left=(bounds?0:(low+max)/(2*max)*100)+'%';mark.style.width=((high-low)/(bounds?max:2*max)*100)+'%';track.append(mark);
  item.append(el('span',{},friendlyLayer(name)),track,el('span',{class:'range-label-value'},bounds?high.toFixed(2):low.toFixed(1)+' / '+high.toFixed(1)));return item;
 }));
 set('range-description',bounds?'Absolute product-sum bounds. Bars are normalized to the largest observed bound; the accumulator limit is 8,388,608.':'Pre-cast minimum / maximum per layer. Bars share a symmetric scale fitted to the observed values, not the full Q12.4 representable range.');
}
const answers={
 host:'Use the bundled C++ runner and independent PyTorch references. This path is verified on the saved 10,000-image test split; the NO_HLS arithmetic is floating point.',
 fixed:'Run the fixed-point C simulation with the supported AMD libraries. The current quantization-aware model and range audit are approximations, and do not certify vendor arithmetic.',
 rtl:'Generate RTL with the supported HLS compiler, then run C/RTL co-simulation with an interface-aware driver. A CPU C++ testbench alone does not drive RTL clocks or AXI transactions.',
 device:'Use synthesis and implementation reports for resource use and timing. Physical I/O, power and real-system behavior still require board/system integration. No physical deployment is claimed here.'
};
function renderDecision(){if($('verification-answer'))set('verification-answer',answers[$('verification-goal').value]);}
function setDataControls(disabled){$$('[data-from-records]').forEach(field=>{field.disabled=disabled;field.setAttribute('aria-busy',String(disabled));});}
async function loadData(){
 const request=++state.request;set('data-status','Loading recorded data…');setDataControls(true);
 if($('retry-data')){$('retry-data').disabled=true;$('retry-data').textContent='Loading…';}
 try{
  const entries=await Promise.all(['metrics','training','ranges','verification'].map(async key=>{
   const link=document.querySelector('link[data-dataset="'+key+'"]');if(!link)throw Error('Missing data source.');
   const response=await fetch(link.href,{cache:'no-cache'});if(!response.ok)throw Error('Data request failed.');
   return [key,await response.json()];
  }));
  const data=validateData(Object.fromEntries(entries));if(request!==state.request)return;
  state.data=data;setDataControls(false);
  if($('data-notice'))$('data-notice').hidden=true;
  initResults();renderTraining();initConfidence();renderRanges();
  set('data-status','Recorded data loaded · 10,000 test images');
  document.documentElement.dataset.resultsState='ready';
  return true;
 }catch(error){
  if(request!==state.request)return false;
  set('data-status','Recorded data unavailable');
  if($('data-notice'))$('data-notice').hidden=false;
  set('data-message',location.protocol==='file:'?'Open the site through an HTTP server to load the recorded result files. The architecture and arithmetic illustrations still work.':'The recorded result files could not be loaded. Check your connection or try reloading the results.');
  document.documentElement.dataset.resultsState='error';
  return false;
 }finally{if(request===state.request&&$('retry-data')){$('retry-data').disabled=false;$('retry-data').textContent='Reload results';}}
}
function init(){
 initTheme();
 on('menu-toggle','click',()=>{const open=$('menu-toggle').getAttribute('aria-expanded')!=='true';$('menu-toggle').setAttribute('aria-expanded',String(open));$('main-nav').classList.toggle('open',open);});
 $$('#main-nav a').forEach(link=>link.addEventListener('click',()=>{$('main-nav').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');}));
 $$('[data-layer]').forEach(button=>button.addEventListener('click',()=>{state.layer=Number(button.dataset.layer);renderLayer();}));renderLayer();
 for(const id of ['tile-layer','tile-size'])on(id,'change',()=>{if($('tile-position'))$('tile-position').value=0;renderTile();});
 on('tile-position','input',renderTile);renderTile();
 on('quant-value','input',renderQuant);on('quant-mode','change',renderQuant);renderQuant();
 on('class-metric','change',renderBars);on('class-sort','change',renderBars);
 on('class-focus','change',()=>{state.focus=Number($('class-focus').value);state.cell=[state.focus,state.focus];updateClassFocus();renderMatrix();});
 $$('[data-matrix-mode]').forEach(button=>button.addEventListener('click',()=>{state.matrix=button.dataset.matrixMode;renderMatrix();}));
 on('confusion-matrix','click',event=>{
  const button=event.target.closest('button[data-row]');if(!button)return;
  state.focus=Number(button.dataset.row);state.cell=[state.focus,Number(button.dataset.col)];
  updateClassFocus();renderMatrix();
  const next=$('confusion-matrix').querySelector('button[data-row="'+state.cell[0]+'"][data-col="'+state.cell[1]+'"]');if(next)next.focus({preventScroll:true});
 });
 on('training-metric','change',renderTraining);on('epoch-position','input',renderTraining);
 on('confidence-threshold','input',renderConfidence);on('range-view','change',renderRanges);
 on('verification-goal','change',renderDecision);renderDecision();
 on('copy-command','click',async()=>{try{if(!navigator.clipboard)throw Error();await navigator.clipboard.writeText($('run-command').textContent);set('copy-status','Command copied.');}catch{set('copy-status','Clipboard unavailable. Select and copy the command above.');}});
 on('retry-data','click',loadData);
 global.siteReady=loadData();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})(typeof window!=='undefined'?window:globalThis);


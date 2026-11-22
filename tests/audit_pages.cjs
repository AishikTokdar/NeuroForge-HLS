// Dependency-free checks for the arithmetic and data used by the UI.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),vm=require('node:vm');
const root=path.resolve(process.argv[2]||path.join(__dirname,'..'));
const api=require(path.join(root,'assets/site.js'));
let checks=0;
function test(name,fn){fn();checks++;console.log('PASS '+name);}
test('Q12.4 ties, saturation and signed representation',()=>{
 for(const [input,raw] of [[.03125,0],[.09375,2],[-.03125,0],[-.09375,-2],[2048,32767],[-2049,-32768]])assert.equal(api.quantize(input).raw,raw);
 assert.equal(api.quantize(-.0625).hex,'0xFFFF');assert.equal(api.quantize(-.0625).bits,'1111111111111111');assert.equal(api.quantize(2048).overflow,true);assert.equal(api.quantize(1).error,0);
});
test('Activation casts floor and wrap at the signed boundary',()=>{
 assert.equal(api.quantize(-.01,'activation').raw,-1);assert.equal(api.quantize(2048,'activation').raw,-32768);assert.equal(api.quantize(-2048.0625,'activation').raw,32767);
 for(const v of [NaN,Infinity,-Infinity,1000001])assert.throws(()=>api.quantize(v));assert.throws(()=>api.quantize(1,'invalid'));
});
test('Tiling counts overlapping reads and excludes external padding',()=>{
 const small=api.tileStats(4,2,1,1);assert.equal(small.tiles,4);assert.equal(small.reads,36);assert.equal(small.buffer,16);
 const whole=api.tileStats(8,8,3,32);assert.equal(whole.tiles,1);assert.equal(whole.reads,192);assert.equal(whole.inputBytes,384);
 const first=api.tileStats(32,8,3,32);assert.equal(first.reads,4332);assert.equal(first.weightBytes,27648);
 assert.throws(()=>api.tileStats(0,8,3,32));assert.throws(()=>api.tileStats(32,0,3,32));
});
test('Architecture parameter and operation totals',()=>{
 assert.equal(api.LAYERS.reduce((n,l)=>n+l.params,0),1439146);assert.equal(api.LAYERS.reduce((n,l)=>n+l.macs,0),53053952);assert.equal(api.LAYERS[4].output,'10 logits');
});
const data=Object.fromEntries(Object.entries({metrics:'classification_metrics',training:'training_metrics',ranges:'numeric_range_audit',verification:'weight_validation'}).map(([k,v])=>[k,JSON.parse(fs.readFileSync(path.join(root,'docs',v+'.json'),'utf8'))]));
test('Recorded results satisfy chart data contracts',()=>{api.validateData(data);assert.equal(data.metrics.confusion_matrix.flat().reduce((a,b)=>a+b,0),10000);assert.equal(data.training.history.length,25);});
test('Incomplete or inconsistent records are rejected',()=>{
 for(const corrupt of [d=>d.metrics.confusion_matrix[0][0]++,d=>d.metrics.per_class[0].recall=1.1,d=>d.training.history[0].loss=null,d=>d.ranges.layers={},d=>d.verification.golden_vector.logits.pop()]){const bad=structuredClone(data);corrupt(bad);assert.throws(()=>api.validateData(bad));}
});
test('First-paint system theme, explicit preference and denied storage',()=>{
 const source=fs.readFileSync(path.join(root,'assets/theme.js'),'utf8');
 for(const [saved,dark,expected] of [[null,true,'dark'],['light',true,'light'],['dark',false,'dark'],['invalid',false,'light']]){const document={documentElement:{dataset:{}}};vm.runInNewContext(source,{document,localStorage:{getItem:()=>saved},window:{matchMedia:()=>({matches:dark})}});assert.equal(document.documentElement.dataset.theme,expected);}
 const document={documentElement:{dataset:{}}};vm.runInNewContext(source,{document,localStorage:{getItem:()=>{throw Error('denied');}},window:{matchMedia:()=>({matches:false})}});assert.equal(document.documentElement.dataset.theme,'light');
});
const files=['index.html',...fs.readdirSync(path.join(root,'visualizations')).filter(n=>n.endsWith('.html')).map(n=>'visualizations/'+n)];assert.equal(files.length,6);
for(const file of files)test(file+' DOM, shared assets and data links',()=>{
 const html=fs.readFileSync(path.join(root,file),'utf8');assert(/<title>[^<]+<\/title>/.test(html));assert(/<html[^>]+lang="en"/.test(html));
 const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(new Set(ids).size,ids.length);
 for(const asset of ['site.css','site.js','theme.js'])assert(html.includes('assets/'+asset));
 for(const dataset of ['metrics','training','ranges','verification'])assert(html.includes('data-dataset="'+dataset+'"'));
 for(const id of ['theme-toggle','theme-system','data-notice','retry-data'])assert(ids.includes(id));
 assert(!/cdn\.jsdelivr|chart\.js|fonts\.googleapis/i.test(html),'Runtime CDN dependency');
 for(const label of html.matchAll(/<label[^>]+for="([^"]+)"/g))assert(ids.includes(label[1]),'Unresolved label');
});
console.log('Pages audit: '+checks+' checks passed across six pages and four saved datasets.');

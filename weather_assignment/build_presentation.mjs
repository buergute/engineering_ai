import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT='/Users/burgut/Downloads/engineering_ai/weather_assignment';
const SKILL='/Users/burgut/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const TMP=path.join(ROOT,'tmp/ppt_build');
const FINAL=path.join(ROOT,'output/姓名_风速预测分析.pptx');
await fs.mkdir(TMP,{recursive:true});
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const report=JSON.parse(await fs.readFile(path.join(ROOT,'output/model_report.json'),'utf8'));
const parseCsv=(s)=>{const lines=s.trim().split(/\r?\n/);const head=lines.shift().split(',');return lines.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[head[i],v])));};
const tests=parseCsv(await fs.readFile(path.join(ROOT,'output/test_predictions.csv'),'utf8'));
const fc=parseCsv(await fs.readFile(path.join(ROOT,'output/forecast_2026-10-10_CNC.csv'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#FFF7E1',title:'#9A3F70',ink:'#222222',blue:'#276DA8',green:'#3B806C',muted:'#5D5D5D',line:'#D9D0B6'};
const font='Hiragino Sans GB';
function box(slide,text,x,y,w,h,size=24,bold=false,color=C.ink){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=text;s.text.style={typeface:font,fontSize:size,bold,color,autoFit:'shrinkText',verticalAlignment:'middle'};return s;
}
function base(title,num){
  const s=p.slides.add();s.background.fill=C.bg;
  box(s,title,65,38,1130,62,36,true,C.title);
  box(s,String(num).padStart(2,'0'),1160,38,60,50,22,false,C.muted);
  return s;
}
function note(slide,text){slide.speakerNotes.textFrame.setText(text);}
function chartFont(chart){applyPresentationChartFont(chart,{fontFamily:font});}
function chart(slide,type,config){const c=slide.charts.add(type,config);chartFont(c);return c;}

// 1: minimal cover, with the source lecture deck's warm palette.
{
  const s=p.slides.add();s.background.fill=C.bg;
  box(s,'Chariton Wind Speed Forecast',82,170,1080,90,54,true,C.title);
  box(s,'10 October 2026 UTC  ·  72 forecasts at 20 minute intervals',84,280,1050,60,29,false,C.ink);
  box(s,'IEM observations  |  Gradient boosting  |  Time ordered backtest',84,375,1080,55,24,false,C.muted);
  box(s,'Course assignment',84,600,500,38,21,false,C.title);
  note(s,'题目来源：用户提供的 c2AIoE2026 课程 PPTX，第35-42页。数据来源：https://mesonet.agron.iastate.edu/request/download.phtml');
}
// 2: source, physical meaning, preprocessing.
{
  const s=base('Data and physical meaning',2);
  box(s,'Source: Iowa Environmental Mesonet, IA_ASOS network, CNC (Chariton)',65,130,1130,45,24,true);
  box(s,'Wind speed (sknt)',70,210,270,40,25,true,C.blue);box(s,'Observed near surface wind in knots; the prediction target',350,210,820,40,23);
  box(s,'Temperature / dew point',70,280,370,40,25,true,C.blue);box(s,'Air mass heat and moisture, both in °F',450,280,720,40,23);
  box(s,'Altimeter pressure',70,350,270,40,25,true,C.blue);box(s,'Pressure in inHg; also use its 12 hour change',350,350,820,40,23);
  box(s,'Preparation',70,455,210,42,25,true,C.title);
  box(s,'Keep UTC observations at :15, :35 and :55. Align to a 20 minute grid. Exclude missing targets from training; the model handles missing inputs. Clip predicted wind at zero knots.',70,510,1100,115,23);
  note(s,'IEM CNC 站页：https://mesonet.agron.iastate.edu/sites/site.php?network=IA_ASOS&station=CNC 。IEM 下载字段与单位：https://mesonet.agron.iastate.edu/request/download.phtml 。原始 CSV 文件位于 weather_assignment/data/CNC_IEM_2024-2026.csv。');
}
// 3: dataset and leakage control.
{
  const s=base('Samples and time ordered split',3);
  box(s,'1 Jan 2024 to 8 Oct 2026 07:15 UTC: 71,838 raw observations',70,130,1110,47,27,true);
  box(s,'Forecast origin is 07:15 each day. Predict all 72 slots two calendar days ahead.',70,218,1110,46,25);
  box(s,'Features use only observations available by origin: current wind, 3/12/24/48 hour summaries, 72/96/168 hour lagged winds, and cyclic target time.',70,285,1100,100,24);
  box(s,'Train',70,460,150,40,25,true,C.blue);box(s,'Origins before 15 Aug 2026',210,460,700,40,23);
  box(s,'Validate',70,515,150,40,25,true,C.green);box(s,'15 Aug to 15 Sep, to select parameters',210,515,780,40,23);
  box(s,'Test',70,570,150,40,25,true,C.title);box(s,'16 Sep to 6 Oct, evaluated once',210,570,700,40,23);
  note(s,'所有切分按预测起点顺序进行。测试集 1415 个有观测值的20分钟时刻。特征构造见 weather_assignment/forecast.py。');
}
// 4: algorithm and settings.
{
  const s=base('Model and parameters',4);
  box(s,'Primary model: direct horizon histogram gradient boosting',70,130,1100,45,27,true);
  box(s,'Shallow trees correct residuals sequentially and capture nonlinear effects of wind history, pressure and hour of day. Direct predictions avoid recursive error growth.',70,205,1100,92,24);
  box(s,'Selected settings',70,350,330,42,25,true,C.title);
  box(s,'180 iterations  |  15 leaves  |  learning rate 0.05  |  min leaf 80  |  L2 = 5',70,410,1110,53,25);
  box(s,'Added multiscale wind summaries, pressure trend and same time lag features. Compared against random forest and a 72 hour lag baseline.',70,530,1090,90,24);
  note(s,'算法：scikit-learn HistGradientBoostingRegressor。具体训练代码与随机种子见 weather_assignment/forecast.py。');
}
// 5: hyperparameter results, native chart and table.
{
  const s=base('Validation: parameter comparison',5);
  const vals=report.validation_results;
  const categories=vals.map(x=>x.model.replaceAll('_',' / '));
  chart(s,'bar',{position:{left:65,top:145,width:800,height:425},categories,series:[{name:'RMSE (knots)',values:vals.map(x=>Number(x.RMSE.toFixed(2))),fill:C.blue}],barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,yAxis:{title:'RMSE (knots)',min:0,max:4,numberFormatCode:'0.0'},dataLabels:{showValue:true,position:'outEnd'}});
  box(s,'Best setting',900,185,300,42,25,true,C.title);
  box(s,'15 leaves\nLearning rate 0.05\nValidation RMSE 3.21 knots',900,250,300,190,24);
  box(s,'More leaves or a higher learning rate increased validation error.',900,500,300,95,21);
  note(s,'数值来自 model_report.json 的 validation_results。验证样本数 2282。RF 配置：100棵树、最大深度12、最小叶样本20。');
}
// 6: test comparison and fitted curve.
{
  const s=base('Test fit and error metrics',6);
  const day=tests.filter(x=>x.target_time.startsWith('2026-10-05') && x.target_time.slice(14,16)==='55');
  chart(s,'line',{position:{left:65,top:145,width:760,height:415},categories:day.map(x=>x.target_time.slice(11,13)),series:[
    {name:'Observed',values:day.map(x=>Number(Number(x.y).toFixed(2))),line:{style:'solid',fill:C.ink,width:3}},
    {name:'Predicted',values:day.map(x=>Number(Number(x.predicted_knots).toFixed(2))),line:{style:'solid',fill:C.blue,width:3}}
  ],hasLegend:true,legend:{position:'bottom'},yAxis:{title:'Wind (knots)',min:0,numberFormatCode:'0.0'}});
  box(s,'Example: 5 Oct 2026 UTC, hourly :55 observations',70,585,760,35,19,false,C.muted);
  const table=s.tables.add({rows:3,columns:3,left:855,top:215,width:365,height:220,values:[
    ['Test model','MAE','RMSE'],
    ['72h lag','2.93','3.79'],
    ['Boosting','2.19','2.84'],
  ]});
  table.styleOptions={headerRow:true,bandedRows:true};
  table.cells.block({row:0,column:0,rowCount:3,columnCount:3}).assign({textStyle:{typeface:font,fontSize:21,color:C.ink}});
  box(s,'Knots, n = 1415',875,465,330,45,20,false,C.muted);
  note(s,'测试集为预测起点 2026-09-16 至 2026-10-06。表格数字来自 model_report.json；图来自 test_predictions.csv。此曲线仅展示测试集中的 2026-10-05 每小时:55。');
}
// 7: residual distribution.
{
  const s=base('Test residual distribution',7);
  const bins=[-10,-8,-6,-4,-2,0,2,4,6,8,10];const counts=Array(bins.length-1).fill(0);
  for(const r of tests){const e=Number(r.error_knots);for(let i=0;i<counts.length;i++){if(e>=bins[i]&&(e<bins[i+1]||i===counts.length-1)){counts[i]++;break;}}}
  chart(s,'bar',{position:{left:70,top:145,width:880,height:445},categories:counts.map((_,i)=>`${bins[i]}~${bins[i+1]}`),series:[{name:'Count',values:counts,fill:C.green}],barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,xAxis:{title:'Predicted minus observed (knots)'},yAxis:{title:'Observations',min:0}});
  box(s,'Mean bias',990,215,230,45,24,true,C.title);
  box(s,'-0.18 knots',990,270,230,55,30,true,C.ink);
  box(s,'The model slightly underpredicts. Extreme winds tend toward the mean.',990,385,210,115,22);
  note(s,'误差分布来自 test_predictions.csv 全测试集。平均偏差来自 model_report.json。');
}
// 8: actual submission forecast.
{
  const s=base('10 October forecast and submission',8);
  const byHour=Array.from({length:24},(_,h)=>Number((fc.slice(h*3,h*3+3).reduce((a,x)=>a+Number(x.forecast_knots),0)/3).toFixed(2)));
  chart(s,'line',{position:{left:65,top:145,width:875,height:430},categories:Array.from({length:24},(_,h)=>String(h).padStart(2,'0')),series:[{name:'Mean of 3 slots',values:byHour,line:{style:'solid',fill:C.blue,width:3}}],hasLegend:false,xAxis:{title:'UTC hour'},yAxis:{title:'Forecast wind (knots)',min:0,max:12,numberFormatCode:'0.0'}});
  box(s,'72 time slots',985,210,225,45,26,true,C.title);
  box(s,`${report.forecast_min.toFixed(1)}–${report.forecast_max.toFixed(1)} knots`,985,265,230,56,26,true,C.ink);
  box(s,'The Excel file retains the original time column and format. Column B holds all 72 forecasts.',985,395,230,130,21);
  box(s,'Chart shows hourly means; the submission retains all 20 minute results.',75,590,1040,38,19,false,C.muted);
  note(s,'预测起点：2026-10-08 07:15 UTC；该日期之后的观测未用于模型。风速单位为 knots，符合用户提供的提交模板表头。IEM 原始观测：https://mesonet.agron.iastate.edu/request/download.phtml');
}

const candidate=path.join(TMP,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:'/Users/burgut/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--require-native-table-slide','6'],
  requiredNativeTableOwnerSlides:[6],requiredNativeChartOwnerSlides:[5,6,7,8],
  fontPolicy:{basis:'design',families:[font]},
  materializeLiteralChartWorkbooks:true,
  receiptPath:path.join(TMP,'validation.json'),
});
console.log(JSON.stringify(result));

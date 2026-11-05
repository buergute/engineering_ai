import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const ROOT='/Users/burgut/Downloads/engineering_ai/weather_assignment';
const SOURCE='/Users/burgut/Downloads/姓名_X.pptx';
const SKILL='/Users/burgut/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const TMP=path.join(ROOT,'tmp/template2/build');
const FINAL=path.join(ROOT,'output/布尔古特_X.pptx');
const ASSETS=path.join(ROOT,'tmp/template2/assets');
await fs.mkdir(TMP,{recursive:true});

const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);

function set(id,value,fontSize){
  const shape=p.resolve(id);
  if(!fontSize){shape.text=value;return;}
  const color=id==='sh/bip8jmho'||id==='sh/w7ulsvqp'?'#C44A4A':'#2A343E';
  shape.text=value.split('\n').map(line=>({runs:[{run:line,style:{fontSize:`${fontSize}px`,typeface:'微软雅黑',color}}]}));
}
async function picture(id,file,alt){
  const image=p.resolve(id);
  const oldFrame=image.frame;
  const oldCrop=image.crop;
  const oldGeometry=image.geometry;
  const bytes=await fs.readFile(path.join(ASSETS,file));
  image.replace({blob:bytes,contentType:'image/png',alt,fit:'contain'});
  image.frame=oldFrame;
  image.crop=oldCrop;
  image.geometry=oldGeometry;
}

for(const id of ['sh/65g3298r','sh/a10jqpsj','sh/3ah8rqlg','sh/ql8jytsj']) {
  p.resolve(id).text='布尔古特';
}

// Slide 1: data, physical meaning, and preprocessing.
set('sh/vmdcj2xw','• IEM CNC 站：2024—2026 年 71,838 条观测\n• 目标：风速 sknt（节）；输入：气温、露点、气压、历史风速',19);
set('sh/dgzetcfa','• 气温/露点表征热湿状态；气压变化反映天气系统。',20);
set('sh/7qp4be9c','• UTC 20 分钟网格对齐，保留 :15、:35、:55 观测；缺失目标不入训练。\n• 构造风速 3/12/24/48 小时统计、72/96/168 小时滞后及时间周期特征。\n• 按预测起点顺序切分；输入只用起点前的观测，避免未来信息泄漏。',18);
set('sh/3qxwn2x8','数据截至 2026-10-08 07:15 UTC；提交结果单位为节（knots）。',18);
set('sh/ozmd8r65','CNC 风速分布',19);
set('sh/5sfepcfe','天气变量与风速关系',19);
await picture('im/ipofexcz','wind_distribution.png','CNC station wind speed histogram');
await picture('im/xofe5sve','wind_relationships.png','Observed temperature and pressure relationships to wind speed');

// Slide 2: method, key parameters, and validation table.
set('sh/nu58f2hs','• 主体：直达目标时刻的直方图梯度提升回归。\n• 18 项历史统计与周期特征，预测两天后 72 个时刻。\n• 180 轮、15 个叶节点、学习率 0.05、最小叶样本 80、L2 正则 5。\n• 采用验证集 RMSE 选参，测试集只用于最终评估。',18);
set('sh/w3i1sfa9','• 固定样本划分，比较叶节点数与学习率。\n• 15 叶 / 0.05 最优；复杂度或学习率增加均未改善验证 RMSE。',19);
set('sh/bip8jmho','验证样本 2,282 个；表中 RMSE 单位为节。',18);
set('sh/gbmxszmt','模型输入与直接预测',19);
set('sh/xk7qlczu','参数对比表',19);
await picture('im/vu5wza18','model_diagram.png','Gradient boosting direct forecast diagram');
const table=p.resolve('tb/sv250jat');
const vals=[['叶节点','学习率','RMSE'],['15','0.05','3.21'],['31','0.05','3.34'],['15','0.10','3.31']];
for(let r=0;r<4;r++)for(let c=0;c<3;c++)table.cells.set(r,c,vals[r][c]);

// Slide 3: task-specific improvements and the untouched test comparison.
set('sh/83ulovat','• 直接预测目标日 72 个时刻，避免递推误差累积。\n\n• 加入多尺度风速统计、气压趋势、跨日同刻滞后特征。\n\n• 与随机森林及 72 小时滞后基线比较。\n\n• 测试集 MAE 2.19 节、RMSE 2.84 节；滞后基线为 2.93 / 3.79 节。\n\n• 未调用外部预报 API。',18);
set('sh/w7ulsvqp','测试集共 1,415 个时刻；测试数据未用于调参。',18);
set('sh/re9g7yxo','测试集误差对比',19);
set('sh/a5c3ql8z','改进模型降低了 MAE 与 RMSE',19);
await picture('im/4repgvex','model_comparison.png','Test MAE and RMSE comparison of baseline and gradient boosting');
const metrics=p.slides.getItem(2).tables.add({
  rows:3,columns:3,left:580,top:500,width:610,height:115,
  values:[
    ['测试模型','MAE（节）','RMSE（节）'],
    ['72h 滞后基线','2.93','3.79'],
    ['梯度提升','2.19','2.84'],
  ],
});
metrics.styleOptions={headerRow:true,bandedRows:true};
metrics.cells.block({row:0,column:0,rowCount:3,columnCount:3}).textStyle.fontSize=16;
metrics.cells.block({row:0,column:0,rowCount:3,columnCount:3}).textStyle.color='#1A354E';
for(let c=0;c<3;c++) metrics.getCell(0,c).fill='#1A354E';
metrics.cells.block({row:0,column:0,rowCount:1,columnCount:3}).textStyle.color='#FFFFFF';
metrics.borders.assign({style:'solid',fill:'#D7E0E5',width:1});

// Slide 4: three actual result figures and concise conclusion.
set('sh/p0batw72','测试集拟合示例',19);
set('sh/oz29krqh','目标日预测曲线',19);
set('sh/3ytsrmpw','测试集残差分布',19);
set('sh/5cva1cfq','测试集 MAE 为 2.19 节、RMSE 为 2.84 节，较 72 小时滞后基线均降低约 25%。平均误差为 -0.18 节，模型略低估，强风波动容易被平滑。10 月 10 日全天 72 个预测值约为 5.82—8.79 节。模型没有使用未来数值天气预报，突发天气变化可能使误差增大。',21);
await picture('im/ozat8bi1','test_fit.png','Held-out test fit for 2 October 2026');
await picture('im/p0ju1gjm','forecast_curve.png','Forecast wind speed for 10 October 2026');
await picture('im/y5cbel0j','residuals.png','Held-out test residual distribution');

const notes=[
  '题目来源：课程作业 PDF，第 1、7、8 页。观测来源：Iowa Environmental Mesonet IA_ASOS / CNC，https://mesonet.agron.iastate.edu/request/download.phtml。原始数据文件：weather_assignment/data/CNC_IEM_2024-2026.csv。风速单位 knots。',
  '模型代码：weather_assignment/forecast.py。验证集起点日期 2026-08-15 至 2026-09-15，样本 2282。表格数据来自 weather_assignment/output/model_report.json。',
  '测试集起点日期 2026-09-16 至 2026-10-06，样本 1415。对比指标来自 model_report.json。未使用外部预报 API。',
  '拟合图来自测试集 2026-10-02；目标日曲线来自 forecast_2026-10-10_CNC.csv；残差图来自 test_predictions.csv。72 个提交值见 布尔古特_X.xlsx。',
];
for(let i=0;i<4;i++)p.slides.getItem(i).speakerNotes.textFrame.setText(notes[i]);

const candidate=path.join(TMP,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:'/Users/burgut/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','4','--require-native-table-slide','2'],
  explicitTotalSlideCount:4,
  requiredNativeTableOwnerSlides:[2],
  receiptPath:path.join(TMP,'validation-named-v2.json'),
});
console.log(JSON.stringify(result));

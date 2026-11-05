import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const input='/Users/burgut/Downloads/姓名_X.pptx';
const output='/Users/burgut/Downloads/engineering_ai/weather_assignment/tmp/template2/roundtrip.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(input));
const q=await p.inspect({kind:'slide,textbox,shape,image,table,layout',maxChars:50000});
await fs.writeFile('/Users/burgut/Downloads/engineering_ai/weather_assignment/tmp/template2/inspect.ndjson',q.ndjson);
console.log('slides',p.slides.items.length);
await (await PresentationFile.exportPptx(p)).save(output);

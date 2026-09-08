const video=document.querySelector('#video');
const caption=document.querySelector('#caption');
const steps=[...document.querySelectorAll('.step')];
const rows=[...document.querySelectorAll('.gantt-row')];
const duration=Number(document.querySelector('.gantt').dataset.duration)||1;
const playhead=document.querySelector('#playhead');
let selected=null;

function selectProcess(index,autoplay=true){
  const step=steps[index];
  if(!step)return;
  selected={index,start:Number(step.dataset.start),end:Number(step.dataset.end)};
  steps.forEach((item,i)=>item.classList.toggle('active',i===index));
  rows.forEach((item,i)=>item.classList.toggle('active',i===index));
  video.currentTime=selected.start;
  caption.textContent=step.querySelector('p').textContent;
  document.querySelector('#number').textContent=index+1;
  document.querySelector('#title').textContent=step.querySelector('b').textContent;
  document.querySelector('#description').textContent=step.querySelector('p').textContent;
  document.querySelector('#duration').textContent=step.querySelector('small').textContent;
  if(autoplay)video.play().catch(()=>{});
}

steps.forEach((item,index)=>item.onclick=()=>selectProcess(index));
rows.forEach((item,index)=>item.onclick=()=>selectProcess(index));
video.addEventListener('timeupdate',()=>{
  const track=rows[0]?.querySelector('.gantt-track');
  if(track)playhead.style.left=`${track.offsetLeft+Math.min(1,video.currentTime/duration)*track.clientWidth}px`;
  if(selected&&video.currentTime>=selected.end-.08){
    video.currentTime=selected.start;
    if(!video.paused)video.play().catch(()=>{});
  }
});
video.addEventListener('ended',()=>{if(selected){video.currentTime=selected.start;video.play().catch(()=>{})}});
if(steps.length)selectProcess(0,false);

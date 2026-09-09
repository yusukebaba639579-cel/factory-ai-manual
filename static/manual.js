const video=document.querySelector('#video'),caption=document.querySelector('#caption');
const steps=[...document.querySelectorAll('.step')],rows=[...document.querySelectorAll('.gantt-row')];
const duration=Number(document.querySelector('.gantt').dataset.duration)||1,playhead=document.querySelector('#playhead');
const playButton=document.querySelector('#play-toggle'),seek=document.querySelector('#seek');
let selected=null;
function timeLabel(value){const minute=Math.floor(value/60),second=Math.floor(value%60);return `${minute}:${String(second).padStart(2,'0')}`}
function selectProcess(index,autoplay=true){const step=steps[index];if(!step)return;selected={index,start:Number(step.dataset.start),end:Number(step.dataset.end)};steps.forEach((item,i)=>item.classList.toggle('active',i===index));rows.forEach((item,i)=>item.classList.toggle('active',i===index));video.currentTime=selected.start;caption.textContent=step.querySelector('p').textContent;document.querySelector('#number').textContent=index+1;document.querySelector('#title').textContent=step.querySelector('b').textContent;document.querySelector('#description').textContent=step.querySelector('p').textContent;document.querySelector('#duration').textContent=step.querySelector('small').textContent;if(autoplay)video.play().catch(()=>{})}
steps.forEach((item,index)=>item.onclick=()=>selectProcess(index));rows.forEach((item,index)=>item.onclick=()=>selectProcess(index));
playButton.onclick=()=>video.paused?video.play():video.pause();
video.addEventListener('play',()=>playButton.textContent='❚❚ 一時停止');video.addEventListener('pause',()=>playButton.textContent='▶ 再生');
seek.oninput=()=>{video.currentTime=Number(seek.value)};
document.querySelectorAll('[data-speed]').forEach(button=>button.onclick=()=>{video.playbackRate=Number(button.dataset.speed);document.querySelectorAll('[data-speed]').forEach(item=>item.classList.toggle('active',item===button))});
document.querySelector('#flip-toggle').onclick=event=>{video.classList.toggle('rotated-video');event.currentTarget.classList.toggle('active')};
video.addEventListener('timeupdate',()=>{seek.value=video.currentTime;document.querySelector('#current-time').textContent=timeLabel(video.currentTime);const track=rows[0]?.querySelector('.gantt-track');if(track)playhead.style.left=`${track.offsetLeft+Math.min(1,video.currentTime/duration)*track.clientWidth}px`;if(selected&&video.currentTime>=selected.end-.08){video.currentTime=selected.start;if(!video.paused)video.play().catch(()=>{})}});
video.addEventListener('ended',()=>{if(selected){video.currentTime=selected.start;video.play().catch(()=>{})}});if(steps.length)selectProcess(0,false);

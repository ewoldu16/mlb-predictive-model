(() => {
  const menu=document.querySelector('.menu'),nav=document.querySelector('#site-nav');
  if(menu&&nav) menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')==='true';menu.setAttribute('aria-expanded',String(!open));nav.classList.toggle('open',!open)});
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const reveal=node=>{node.classList.add('is-visible');node.querySelectorAll('[data-bar]').forEach(bar=>bar.style.width=`${Math.max(0,Math.min(100,Number(bar.dataset.bar)||0))}%`)};
  const nodes=[...document.querySelectorAll('[data-reveal]')];
  if(reduced||!('IntersectionObserver' in window)) nodes.forEach(reveal); else {const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){reveal(entry.target);observer.unobserve(entry.target)}}),{threshold:.08});nodes.forEach(node=>observer.observe(node))}
})();

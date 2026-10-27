function sizeDemo(frame) {
  try {
    const body = frame.contentDocument.body;
    const style = frame.contentWindow.getComputedStyle(body);
    const height = Math.ceil(body.getBoundingClientRect().height + parseFloat(style.marginTop) + parseFloat(style.marginBottom));
    if (Math.abs(parseFloat(frame.style.height || '0') - height) > 2) frame.style.height = height + 'px';
  } catch (_) {}
}
document.querySelectorAll('iframe.demo').forEach(frame => {
  function observeDemo() {
    sizeDemo(frame);
    try { new ResizeObserver(() => sizeDemo(frame)).observe(frame.contentDocument.body); } catch (_) {}
  }
  frame.addEventListener('load', observeDemo);
  if (frame.contentDocument?.readyState === 'complete') observeDemo();
});
if (window.matchMedia('(max-width: 950px)').matches) document.querySelector('aside details')?.removeAttribute('open');

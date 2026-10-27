document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.math').forEach(node => {
    if (window.katex) katex.render(node.textContent, node, {displayMode:true,throwOnError:false});
  });
  if (window.mermaid) {
    mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'neutral'});
    await mermaid.run({querySelector:'.mermaid'});
  }
});
function sizeDemo(frame) {
  try { frame.style.height = (frame.contentDocument.documentElement.scrollHeight + 32) + 'px'; } catch (_) {}
}
document.querySelectorAll('iframe.demo').forEach(frame => {
  frame.addEventListener('load', () => {
    sizeDemo(frame);
    try { new ResizeObserver(() => sizeDemo(frame)).observe(frame.contentDocument.body); } catch (_) {}
  });
});

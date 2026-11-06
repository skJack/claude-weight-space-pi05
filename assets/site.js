(function () {
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Video gallery: one player, one tab per clip.
  document.querySelectorAll("[data-gallery]").forEach(function (root) {
    var video = root.querySelector("video");
    var prompt = root.querySelector(".clip-prompt");
    var meta = root.querySelector(".clip-meta");
    var tabs = root.querySelectorAll(".tab");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) { t.setAttribute("aria-pressed", t === tab ? "true" : "false"); });
        video.poster = tab.dataset.poster;
        video.src = tab.dataset.src;
        prompt.textContent = tab.dataset.prompt;
        meta.textContent = tab.dataset.meta;
        if (!reduced) {
          var p = video.play();
          if (p && p.catch) { p.catch(function () {}); }
        }
      });
    });
    if ("IntersectionObserver" in window && !reduced) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            var p = video.play();
            if (p && p.catch) { p.catch(function () {}); }
          } else {
            video.pause();
          }
        });
      }, { threshold: 0.4 }).observe(video);
    }
  });

  // Image compare: drag the divider between the raw render and the training frame.
  document.querySelectorAll("[data-compare]").forEach(function (root) {
    var stage = root.querySelector(".stage.slider");
    var range = root.querySelector("input[type=range]");
    var stages = root.querySelectorAll(".stage");
    var top = root.querySelector(".top");
    var base = root.querySelector(".base");
    var diff = root.querySelector(".diff");
    var caption = root.querySelector(".pair-caption");
    var zoom = root.querySelector(".zoom-toggle input");
    var buttons = root.querySelectorAll(".tab");
    function setOrigin(x, y) {
      root.querySelectorAll(".zoomer").forEach(function (z) { z.style.transformOrigin = x + "% " + y + "%"; });
    }
    range.addEventListener("input", function () { stage.style.setProperty("--pos", range.value + "%"); });
    zoom.addEventListener("change", function () {
      stages.forEach(function (s) { s.classList.toggle("zoomed", zoom.checked); });
    });
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        buttons.forEach(function (t) { t.setAttribute("aria-pressed", t === b ? "true" : "false"); });
        top.src = b.dataset.raw;
        base.src = b.dataset.mp4;
        diff.src = b.dataset.diff;
        caption.textContent = b.dataset.caption;
        setOrigin(b.dataset.x, b.dataset.y);
      });
    });
    var first = root.querySelector('.tab[aria-pressed="true"]');
    if (first) { setOrigin(first.dataset.x, first.dataset.y); }
  });

  // Copy the BibTeX entry.
  document.querySelectorAll("[data-copy]").forEach(function (root) {
    var btn = root.querySelector("button");
    var label = btn.textContent;
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(root.querySelector("code").textContent).then(function () {
        btn.textContent = root.dataset.copied;
        setTimeout(function () { btn.textContent = label; }, 1500);
      });
    });
  });

  // Line chart hover: nearest plan window, all four values.
  document.querySelectorAll("[data-linechart]").forEach(function (root) {
    var data = JSON.parse(root.dataset.linechart);
    var svg = root.querySelector("svg");
    var tip = root.querySelector(".tooltip");
    var guide = svg.querySelector(".guide");
    function hide() { tip.style.display = "none"; guide.setAttribute("opacity", "0"); }
    function show(evt) {
      var r = svg.getBoundingClientRect();
      var sx = data.vbw / r.width;
      var x = (evt.clientX - r.left) * sx;
      if (x < data.x0 - 12 || x > data.x1 + 12) { hide(); return; }
      var best = 0, dist = 1e9;
      data.xs.forEach(function (v, i) {
        var d = Math.abs(v - x);
        if (d < dist) { dist = d; best = i; }
      });
      guide.setAttribute("x1", data.xs[best]);
      guide.setAttribute("x2", data.xs[best]);
      guide.setAttribute("opacity", "1");
      var html = "<b>" + data.windows[best] + "</b>";
      data.series.forEach(function (s) {
        html += '<div class="row"><span><i class="key ' + s.key + '"></i>' + s.label + "</span><span>" + s.values[best].toFixed(3) + "</span></div>";
      });
      tip.innerHTML = html;
      tip.style.display = "block";
      var px = data.xs[best] / sx;
      var w = tip.offsetWidth;
      tip.style.left = (px + 14 + w > r.width ? px - w - 14 : px + 14) + "px";
    }
    svg.addEventListener("mousemove", show);
    svg.addEventListener("mouseleave", hide);
    svg.addEventListener("touchstart", function (e) { show(e.touches[0]); }, { passive: true });
    svg.addEventListener("touchmove", function (e) { show(e.touches[0]); }, { passive: true });
    svg.addEventListener("touchend", hide);
  });
})();

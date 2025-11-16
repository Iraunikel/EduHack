const demoData = {
  metrics: {
    totalFiles: 16,
    totalChunks: 42,
    languages: {
      unknown: 8,
      en: 2,
      fr: 2,
      de: 2,
      lb: 2,
    },
    vectorDbPath: "output/chroma_db",
    evaluation: {
      tokenReductionPercent: 92.21,
      latencyReductionPercent: 46.96,
      relevanceIncrease: 0.26,
    },
  },
};

const detectorLeftHtml =
  "<p><strong>Input folder:</strong> tests/test_data</p>" +
  "<p><strong>Example file:</strong> feedback_en_comprehensive.docx</p>" +
  "<p>Real student feedback about a course in multiple languages (EN, FR, DE, LB).</p>";

const optimizerLeftHtml =
  "<p><strong>Before optimization</strong></p>" +
  "<p>Too much theory, not enough practice... (raw text, separators, duplicates).</p>";

const embeddingsLeftHtml =
  "<p><strong>Optimized chunks prepared for embeddings</strong></p>" +
  "<div class=\"chunk-card\"><div class=\"chunk-meta\">EN · docx</div>" +
  "<div class=\"chunk-text\">" +
  "Too much theory, not enough practice. The course structure could be better organized. " +
  "We would like more opportunities for interaction. The concepts are well explained " +
  "but practical application is limited." +
  "</div></div>" +
  "<div class=\"chunk-card\"><div class=\"chunk-meta\">FR · pdf</div>" +
  "<div class=\"chunk-text\">" +
  "Trop de théorie, pas assez de pratique. La structure du cours pourrait être mieux organisée. " +
  "Nous aurions besoin de plus de ressources complémentaires. Nous aimerions plus d'opportunités d'interaction." +
  "</div></div>" +
  "<div class=\"chunk-card\"><div class=\"chunk-meta\">LB · pdf</div>" +
  "<div class=\"chunk-text\">" +
  "Ze vill Theorie, ze wéineg Praxis. D'Struktur vum Cours kéint besser organiséiert ginn. " +
  "Mir géifen méi ergänzend Ressourcen brauchen. Mir géifen gär méi Méiglechkeete fir Interaktioun hunn." +
  "</div></div>";

const evaluatorLeftHtml =
  "<p><strong>Evaluation query</strong></p>" +
  "<p>\"What are the main issues students mention in their feedback?\"</p>" +
  "<p><strong>Ground truth summary</strong></p>" +
  "<p>Too much theory, not enough practice, need for more interaction, better structure, " +
  "updated materials, and higher engagement.</p>";


function getElement(id) {
  // Get element by id.
  return document.getElementById(id);
}


function setStep(step) {
  // Switch the active step and update labels.
  const buttons = document.querySelectorAll(".step-button");
  let index = 0;
  while (index < buttons.length) {
    const button = buttons[index];
    const buttonStep = button.getAttribute("data-step");
    if (buttonStep === step) {
      button.classList.add("active");
    } else {
      button.classList.remove("active");
    }
    index += 1;
  }
  const titleElement = getElement("panel-title");
  const actionButton = getElement("action-button");
  if (step === "1") {
    titleElement.textContent = "Raw input - course feedback document";
    actionButton.textContent = "Run Detector";
  } else if (step === "2") {
    titleElement.textContent = "Detected issues and prepared text";
    actionButton.textContent = "Run Optimizer";
  } else if (step === "3") {
    titleElement.textContent = "Optimized chunks ready for embeddings";
    actionButton.textContent = "Generate Embeddings";
  } else {
    titleElement.textContent = "RAG-style query and evaluation";
    actionButton.textContent = "Run Evaluation";
  }
}


function renderContent(step) {
  // Render left and right panels for a given step.
  const leftBody = getElement("panel-left-body");
  const rightBody = getElement("panel-right-body");
  if (step === "1") {
    const metrics = demoData.metrics;
    const langs = metrics.languages;
    leftBody.innerHTML = detectorLeftHtml;
    rightBody.innerHTML =
      "<div class=\"metric-row\"><span class=\"metric-label\">Files</span>" +
      "<span class=\"metric-value\">" +
      metrics.totalFiles +
      "</span></div>" +
      "<div class=\"metric-row\"><span class=\"metric-label\">Chunks</span>" +
      "<span class=\"metric-value\">" +
      metrics.totalChunks +
      "</span></div>" +
      "<div class=\"metric-row\"><span class=\"metric-label\">Languages</span>" +
      "<span class=\"metric-value\">EN, FR, DE, LB</span></div>" +
      "<p style=\"margin-top:8px;\">Detector scans the folder, identifies file types and languages, " +
      "and prepares structured metadata for the other agents.</p>" +
      "<p><small>Language counts: " +
      JSON.stringify(langs) +
      "</small></p>";
  } else if (step === "2") {
    leftBody.innerHTML = optimizerLeftHtml;
    rightBody.innerHTML =
      "<p><strong>After optimization</strong></p>" +
      "<p>Duplicates are removed (over 15K lines), text is cleaned, and the dataset is " +
      "split into 42 semantically coherent chunks across all languages.</p>";
  } else if (step === "3") {
    leftBody.innerHTML = embeddingsLeftHtml;
    rightBody.innerHTML =
      "<p><strong>Embeddings stored locally</strong></p>" +
      "<p>Each chunk is converted into a multilingual vector and stored in ChromaDB at:</p>" +
      "<p><code>" +
      demoData.metrics.vectorDbPath +
      "</code></p>" +
      "<p>Chunks that talk about \"too much theory\" cluster together even across EN, FR, DE, and LB.</p>" +
      "<div class=\"cluster-legend\">Conceptual view: each dot = one chunk; color = language; cluster = topic.</div>";
  } else {
    const evalData = demoData.metrics.evaluation;
    leftBody.innerHTML = evaluatorLeftHtml;
    rightBody.innerHTML =
      "<div class=\"scorecard\">" +
      "<div class=\"scorecard-header\">" +
      "<span class=\"score-label\">Token reduction</span>" +
      "<span class=\"score-value\">" +
      evalData.tokenReductionPercent.toFixed(2) +
      "%</span></div>" +
      "<div class=\"scorecard-header\">" +
      "<span class=\"score-label\">Latency reduction</span>" +
      "<span class=\"score-value\">" +
      evalData.latencyReductionPercent.toFixed(2) +
      "%</span></div>" +
      "<div class=\"scorecard-header\">" +
      "<span class=\"score-label\">Relevance increase</span>" +
      "<span class=\"score-value\">+" +
      evalData.relevanceIncrease.toFixed(2) +
      "</span></div>" +
      "<div class=\"improvement\">" +
      "Evaluator compares a raw LLM call against a RAG-style call over optimized chunks " +
      "and shows that we use ~92% fewer tokens, almost half the latency, and higher semantic relevance." +
      "</div></div>";
  }
}


function bindEvents() {
  // Bind step buttons and action button.
  const buttons = document.querySelectorAll(".step-button");
  let index = 0;
  while (index < buttons.length) {
    const button = buttons[index];
    button.addEventListener("click", function handleClick() {
      const step = button.getAttribute("data-step");
      setStep(step);
      renderContent(step);
    });
    index += 1;
  }
  const actionButton = getElement("action-button");
  actionButton.addEventListener("click", function handleAction() {
    const activeButton = document.querySelector(".step-button.active");
    const step = activeButton.getAttribute("data-step");
    renderContent(step);
  });
}


function initDemo() {
  // Initialize demo handlers and default view.
  const statusText = getElement("status-text");
  statusText.textContent =
    "Files: 16 · Chunks: 42 · Languages: EN, FR, DE, LB";
  setStep("1");
  renderContent("1");
  bindEvents();
}


window.addEventListener("DOMContentLoaded", initDemo);



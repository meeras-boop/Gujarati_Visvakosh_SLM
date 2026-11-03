"""Gujarati Vishwakosh research showcase: static, no model loading/training."""
import re
import unicodedata
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Gujarati Vishwakosh | SLM Research',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Gujarati:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp {font-family:Inter,'Noto Sans Gujarati',sans-serif}
.stApp {background:linear-gradient(145deg,#f7f9ff 0%,#fff 60%,#f4f8ff 100%);color:#18253e}
[data-testid="stSidebar"] {background:#10223d} [data-testid="stSidebar"] * {color:#f2f6ff!important}
.hero {padding:2.3rem 2.5rem;border-radius:22px;background:linear-gradient(115deg,#142d53,#264e82);color:white;margin-bottom:1.3rem;box-shadow:0 12px 35px #173b6920}
.hero h1 {font-size:2.1rem;margin:0 0 .6rem;color:white}.hero p {color:#dce9ff;margin:0}
.panel {background:white;border:1px solid #e3eaf3;border-radius:16px;padding:1.1rem 1.3rem;margin:.5rem 0 1rem;box-shadow:0 3px 14px #172f510a}
.step {border-left:4px solid #4975c7;padding:.75rem 1rem;background:white;border-radius:9px;margin:.55rem 0}
.small {font-size:.86rem;color:#64748b}.chip {display:inline-block;background:#eaf2ff;color:#17457d;border-radius:20px;padding:.3rem .75rem;margin:.2rem;font-size:.83rem}
</style>''',unsafe_allow_html=True)

FEATURES={
'G01':'Unicode NFC normalization','G22':'Repeated whitespace cleanup','G12':'Sentence punctuation consistency',
'G13':'Comma and punctuation spacing','G25':'Numeric Wikipedia citation markers','G42':'Reviewed Gujarati terminology',
'G48':'Terminology consistency','G91':'Mean sentence length','G96':'Long sentence detection',
'G97':'Safe sentence restructuring review','G44':'Colloquial-language review','G90':'Encyclopedic declarative prose',
'G146':'Formal register','G147':'Encyclopedic tone','G167':'Numerical information retention',
'G124':'Paragraph and information retention'}
LOSSES=pd.DataFrame({'Epoch':[1,2,3,4,5], 'Training loss':[0.799520,0.504488,0.456098,0.428897,0.411253], 'Validation loss':[0.455129,0.443797,0.439975,0.437026,0.433460]})
CITATION=re.compile(r'\[\s*[0-9૦-૯]+(?:\s*[,;–-]\s*[0-9૦-૯]+)*\s*\]')
WIKI_LINK=re.compile(r'\[\[([^\]|]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]')
NUMBERS=re.compile(r'[0-9૦-૯]+(?:[.,][0-9૦-૯]+)*')
DEFAULT_TERMS={'ટ્યુરિંગ એવોર્ડ':'ટ્યુરિંગ પુરસ્કાર','કમ્પ્યુટર':'કમ્પ્યૂટર','સોફ્ટવેર':'સૉફ્ટવેર'}
EXAMPLE='જ્યોફ્રી હિન્ટન કૃત્રિમ ચેતાતંત્રો પરના સંશોધન માટે જાણીતા છે. [12] તેમણે 2018માં ટ્યુરિંગ એવોર્ડ મેળવ્યો હતો. [13]'

def static_edit(source, remove_citations=True, normalize_spacing=True, apply_terms=False, terms=None):
    """Conservative deterministic cleanup; does not claim semantic rewriting."""
    t=unicodedata.normalize('NFC',source)
    changes=[]
    if '[[' in t:
        u=WIKI_LINK.sub(lambda m:m.group(2) or m.group(3) or '',t)
        if u!=t: changes.append('Wikipedia internal link markup converted to display text')
        t=u
    if remove_citations:
        u=CITATION.sub('',t)
        if u!=t: changes.append('Bracketed numeric citations removed')
        t=u
    if normalize_spacing:
        u=t.replace('\u00a0',' ')
        u=re.sub(r'[ \t]+',' ',u)
        u=re.sub(r'\s+([,;:।.!?])',r'\1',u)
        u=re.sub(r'([,;:])(?=[^\s\d])',r'\1 ',u)
        u=re.sub(r' *\n *','\n',u)
        u=re.sub(r'\n{3,}','\n\n',u).strip()
        if u!=t: changes.append('Whitespace and punctuation spacing normalized')
        t=u
    if apply_terms and terms:
        for a,b in sorted(terms.items(),key=lambda kv:-len(kv[0])):
            if a in t and a!=b:
                t=t.replace(a,b)
                changes.append(f'Reviewed term substitution: {a} → {b}')
    return t,changes

def text_metrics(t):
    s=[x.strip() for x in re.split(r'(?<=[.!?।])\s+',t) if x.strip()]
    words=t.split()
    return {'Characters':len(t),'Words':len(words),'Sentences':len(s),'Mean words / sentence':round(sum(len(x.split()) for x in s)/len(s),2) if s else 0,'Citation markers':len(CITATION.findall(t)),'Repeated spaces':len(re.findall(r' {2,}',t)),'Paragraphs':len([x for x in re.split(r'\n\s*\n',t) if x.strip()])}

def pipeline(items):
    for i,(title,detail) in enumerate(items,1):
        st.markdown(f'<div class="step"><b>{i:02d} · {title}</b><div class="small">{detail}</div></div>',unsafe_allow_html=True)
        if i<len(items): st.markdown('<div style="margin-left:24px;color:#4975c7;font-size:1.2rem">↓</div>',unsafe_allow_html=True)

st.sidebar.markdown('## 📚 Gujarati Vishwakosh')
st.sidebar.caption('Research showcase · Static demonstration')
page=st.sidebar.radio('Explore', ['Overview','Live static editor','Research journey','Final ByT5 pipeline','Training analytics','Model comparison','Style feature matrix','Evaluation & limitations','Deployment guide'])
st.sidebar.divider()
st.sidebar.info('This application never loads or trains an SLM. The editor is deterministic; training figures are documented experiment results.')

if page=='Overview':
    st.markdown('<div class="hero"><h1>Gujarati Vishwakosh · Style-Controlled SLM Research</h1><p>Gujarati encyclopedic editing • multilingual small language models • transparent, fact-preserving workflows</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('Source articles',44);b.metric('Converted references',43);c.metric('Candidate training pairs',1461);d.metric('Final validation loss', '0.4335')
    st.subheader('Research objective')
    st.write('Study Gujarati Vishwakosh editorial conventions and develop an editor that transforms existing Wikipedia-style Gujarati into formal encyclopedic Gujarati while preserving facts, dates, names, and meaning. The final reported trained-model experiment uses **google/byt5-small**. This public demonstration intentionally uses **static rules only**.')
    st.markdown('<span class="chip">Gujarati NLP</span><span class="chip">Text-to-text editing</span><span class="chip">ByT5</span><span class="chip">mT5</span><span class="chip">Evidence-based evaluation</span>',unsafe_allow_html=True)
    st.subheader('Research flow at a glance')
    pipeline([('Corpus preparation','Wikipedia articles + converted Vishwakosh-style references + Excel reference'),('Style analysis','Editorial features, measurable diagnostics, protected facts'),('Model exploration','Rule-based teacher, mT5 experiments, ByT5 experiments'),('Final trained experiment','ByT5-small; manual PyTorch training, 5 epochs'),('Evaluation','Held-out loss, generation checks, 20 Gujarati tests; human fidelity review still required'),('Presentation-only deployment','Static Streamlit editor with no model weights or GPU')])

elif page=='Live static editor':
    st.title('✍️ Live Gujarati Vishwakosh-style editor')
    st.warning('Static editorial formatting only. Formal tone and meaning-preserving paraphrasing cannot be guaranteed without reviewed rules or a model. No model is loaded.')
    src=st.text_area('Input Gujarati paragraph',value=EXAMPLE,height=180)
    c1,c2,c3=st.columns(3)
    citation=c1.checkbox('Remove numeric citations',value=True)
    spacing=c2.checkbox('Normalize spacing',value=True)
    terms_on=c3.checkbox('Apply reviewed terminology',value=False)
    with st.expander('Terminology substitutions — editable and opt-in'):
        st.caption('Illustrative mappings, not validated against the entire Vishwakosh corpus. Review each mapping before using it.')
        raw_terms=st.text_area('One source=target mapping per line',value='\n'.join(f'{a}={b}' for a,b in DEFAULT_TERMS.items()),height=120)
    terms={}
    for line in raw_terms.splitlines():
        if '=' in line:
            a,b=line.split('=',1)
            if a.strip() and b.strip():terms[a.strip()]=b.strip()
    output,changes=static_edit(src,citation,spacing,terms_on,terms)
    l,r=st.columns(2)
    with l:
        st.markdown('#### Original')
        st.text_area('Original text',src,height=220,disabled=True,label_visibility='collapsed')
    with r:
        st.markdown('#### Edited')
        st.text_area('Edited text',output,height=220,label_visibility='collapsed')
    st.download_button('Download edited Gujarati text',output,file_name='vishwakosh_static_edit.txt',mime='text/plain')
    if changes: st.success('Applied: '+'; '.join(changes))
    else: st.info('No eligible deterministic changes were found.')
    if Counter(NUMBERS.findall(src))!=Counter(NUMBERS.findall(output)):
        st.error('Numeric content changed. Review before using this output.')
    st.subheader('Before / after diagnostics')
    metrics=pd.DataFrame({'Before':text_metrics(src),'After':text_metrics(output)})
    st.dataframe(metrics,use_container_width=True)
    st.caption('Passing numeric and length checks does not establish semantic equivalence. Human review remains necessary.')

elif page=='Research journey':
    st.title('🧪 Research journey and experiment progression')
    st.write('The following stages distinguish documented experiments from ideas and unverified outcomes. Not every listed model has a verified successful training run.')
    stages=[
      ('Stage A · Comparative style study','Wikipedia versus Gujarati Vishwakosh examples','Defined general/specific stylistic characteristics; 310-feature research inventory requested; 16 practical features selected for ByT5 V4/V5.','Research/design'),
      ('Stage B · Deterministic teacher','CLEAN / FORMALIZE / SHORTEN / COPY','mT5-style workflow used citation cleanup, whitespace normalization, verified substitutions, conservative splitting and copy targets. Authentic passage pairs were disabled in the supplied teacher code.','Implemented code; training outcome not established here'),
      ('Stage C · mT5-small','google/mt5-small','Multilingual encoder–decoder editing experiment; code includes FP32 loss diagnostics and guarded inference. Previous runs experienced invalid losses or poor generations.','Experimental; no validated successful model claimed'),
      ('Stage D · Qwen exploration','Qwen 2.5 1.5B / 7B','Instruction-tuned style rewriting and QLoRA exploration in earlier Colab work; low/zero reported accuracy and training issues motivated further debugging.','Exploratory; do not infer successful training'),
      ('Stage E · ByT5 with Trainer','google/byt5-small · V4/V4.1','1,096 aligned conversion candidates + 365 cleanup examples; Trainer path produced NaN despite finite pre-training forward/backward checks.','Failed numerical training path'),
      ('Stage F · ByT5 manual PyTorch','google/byt5-small · V5','Three manual optimizer steps and then 20-step preflight passed; 6,000 training steps across five epochs; best validation loss 0.433460.','Completed training; final reported model'),
      ('Stage G · Static Streamlit showcase','No weights, no training, no inference','Deterministic editorial cleanup and interactive PhD research presentation.','Live application demonstration')]
    for title,tech,description,status in stages:
        with st.expander(title,expanded=('Stage F' in title)):
            st.markdown(f'**Approach:** {tech}')
            st.write(description)
            st.caption('Evidence status: '+status)
    st.subheader('Illustrative evolution')
    pipeline([('Original Wikipedia Gujarati','Source sentence with citation/markup noise'),('Rule-based supervision','Example: “વાક્ય. [12]” → “વાક્ય.”'),('Multilingual encoder–decoder trials','mT5-small; ByT5-small; some paths unstable'),('Manual optimizer repair','FP32, AdamW, clipped gradients, finite-value checks'),('Final ByT5 checkpoint','Saved after epoch 5; generation quality must still be audited')])

elif page=='Final ByT5 pipeline':
    st.title('🏁 Final trained model · ByT5-small V5')
    st.success('Reported training completed: 5 epochs / 6,000 updates / validation loss 0.433460. This page does not load the checkpoint.')
    pipeline([('Existing Google Drive data','44 Wikipedia articles; 43 converted articles; Vishwakosh Excel reference'),('Title matching','43 matched article titles'),('Candidate alignment','1,096 approximate sentence-level converted pairs; not human-verified'),('Deterministic cleanup examples','365 synthetic cleanup pairs; 1,461 total candidates'),('Article-level split','1,200 training + 160 validation examples; leakage reduced by title grouping'),('Byte-level tokenization','ByT5Tokenizer; short input/output spans, 512 token limit'),('20-step numerical preflight','Manual FP32 forward/backward; gradient clipping; reject non-finite values'),('Manual PyTorch fine-tuning','ByT5-small; AdamW; LR 1e-5; batch 1; 5 epochs; 6,000 updates'),('Validation and checkpoint','Validation loss each epoch; best checkpoint saved to Google Drive'),('Constrained generation and audit','Check numeric retention, overlap, length, Gujarati script; fallback to deterministic editor'),('20-topic Gujarati evaluation','Raw model outputs and final guarded outputs; human review required')])
    st.subheader('Architecture schematic')
    st.graphviz_chart('''digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fillcolor="#eaf2ff" color="#4475b8" fontname="Helvetica" fontsize=11]; A[label="Gujarati source"]; B[label="ByT5 encoder"]; C[label="ByT5 decoder"]; D[label="Raw rewrite"]; E[label="Safety audit" fillcolor="#fff0d6"]; F[label="Accepted output" fillcolor="#dff5e8"]; G[label="Static fallback" fillcolor="#f8e5e5"]; A->B->C->D->E; E->F[label="pass"]; E->G[label="reject"]; }''',use_container_width=True)
    st.code('MyDrive/Gujarati_Visvakosh_ByT5_Style_Editor_V5_Manual/model',language='text')
    st.caption('Checkpoint location is a reported Drive path; Streamlit does not access it.')

elif page=='Training analytics':
    st.title('📉 Training analytics · final V5 run')
    a,b,c=st.columns(3)
    a.metric('Epochs','5');b.metric('Updates','6,000');c.metric('Best epoch','5')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=LOSSES['Epoch'],y=LOSSES['Training loss'],mode='lines+markers',name='Training loss',line=dict(width=3,color='#3366b0')))
    fig.add_trace(go.Scatter(x=LOSSES['Epoch'],y=LOSSES['Validation loss'],mode='lines+markers',name='Validation loss',line=dict(width=3,color='#d28b3b')))
    fig.update_layout(template='plotly_white',title='Epoch-level loss',xaxis_title='Epoch',yaxis_title='Loss',height=440,legend=dict(orientation='h',y=1.1),margin=dict(l=20,r=20,t=65,b=30))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(LOSSES.style.format({'Training loss':'{:.6f}','Validation loss':'{:.6f}'}),hide_index=True,use_container_width=True)
    st.download_button('Download loss history CSV',LOSSES.to_csv(index=False).encode('utf-8'),file_name='byt5_v5_loss_history.csv',mime='text/csv')
    st.markdown('#### Interpretation')
    st.write('Training loss declined from **0.799520 to 0.411253**. Validation loss declined from **0.455129 to 0.433460**. No NaN was reported in the completed manual training run. The decrease supports successful optimization, **not** a claim of factual correctness or human-rated Vishwakosh style.')
    st.caption('Values transcribed from the supplied final V5 run. These are not live training measurements.')

elif page=='Model comparison':
    st.title('🔬 Model and method comparison')
    rows=[
      ['Deterministic editor','No learned parameters','Implemented','Citation/spacing cleanup; limited terminology','Does not learn broad style'],
      ['mT5-small','Multilingual encoder–decoder','Explored / diagnostic','Synthetic operation-based editing','Reliable final training not established'],
      ['Qwen 2.5 1.5B','Instruction-tuned causal LM','Explored','Prompt/QLoRA style adaptation','Earlier poor/zero accuracy reports'],
      ['Qwen 2.5 7B','Instruction-tuned causal LM','Explored','Larger model adaptation attempt','No verified successful final outcome'],
      ['ByT5-small V4.1','Multilingual byte-level seq2seq','Failed Trainer path','Forward/backward finite before training','NaN at first Trainer step'],
      ['ByT5-small V5','Multilingual byte-level seq2seq','Completed training','Manual PyTorch; 5 epochs; val loss 0.433460','Observed omissions/repetition; needs fidelity evaluation']]
    st.dataframe(pd.DataFrame(rows,columns=['System','Architecture','Evidence status','Contribution','Limitation']),hide_index=True,use_container_width=True)
    st.info('“Explored” is not synonymous with “trained successfully.” No accuracy or F1 values are invented for models without verified evaluations.')

elif page=='Style feature matrix':
    st.title('🧩 Sixteen selected editorial features')
    st.write('These are the 16 practical features embedded in the ByT5 workflow. A broader 310-feature inventory was requested in the research program, but its measurements are not validated in this app.')
    df=pd.DataFrame([{'Code':k,'Feature':v,'Static editor support':('Automated' if k in ['G01','G22','G12','G13','G25'] else 'Optional reviewed mapping' if k in ['G42','G48'] else 'Diagnostic / human review')} for k,v in FEATURES.items()])
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.download_button('Download 16-feature matrix',df.to_csv(index=False).encode('utf-8'),file_name='selected_style_features.csv',mime='text/csv')
    st.markdown('#### Editorial principles')
    st.write('Preserve chronology, dates, numbers, technical terms, negation, named entities and all source facts. Avoid adding unsupported information. A stylistic preference must never justify deleting facts.')

elif page=='Evaluation & limitations':
    st.title('🛡️ Evaluation, risks, and research integrity')
    st.subheader('Observed error cases')
    examples=[
      ('Semantic drift','ભૂગર્ભજળ એ જમીનની નીચે આવેલા ...','ભૂગર્ભજળની નીચે આવેલા ...','“Below ground” became “below groundwater”; meaning changed.'),
      ('Unnecessary repetition','જૂનાં પુસ્તકોનું સંરક્ષણ ગ્રંથાલય અને સંશોધન સંસ્થાઓ માટે ...','જૂનાં પુસ્તકોનું સંરક્ષણ ગ્રંથાલય અને સંશોધન સંસ્થાઓનું સંરક્ષણ ગ્રંથાલય અને સંશોધન સંસ્થાઓ માટે ...','Repeated a phrase while still passing simple automatic checks.'),
      ('Factual omission','પછી પીગળેલી ધાતુની ચોક્કસ માત્રા ... ગરમ ન કરેલા શોટ ચેમ્બર ...','પીગળેલી ધાતુની ચોક્કસ માત્રા કોલ્ડ-ચેમ્બર મશીનમાં પરિવહન કરવામાં આવે છે.','Dropped the unheated shot chamber / injection cylinder detail; guard rejected.')]
    for heading,source,outcome,note in examples:
        with st.expander(heading,expanded=True):
            st.markdown('**Source:** '+source)
            st.markdown('**Generated:** '+outcome)
            st.error(note)
    st.subheader('Suggested PhD evaluation protocol')
    st.write('On held-out article-level test examples, record source, reference, raw generation, final guarded output and acceptance status. Ask two Gujarati-fluent annotators to rate factual preservation, grammar, and Vishwakosh register independently. Report agreement (e.g. Cohen’s κ for categorical ratings), accepted-edit rate, unchanged/copy rate, omission rate, repetition rate, numeric retention, and per-category style measurements. Distinguish raw-model quality from fallback-assisted quality.')
    st.warning('A low loss is not style-transfer accuracy. Numeric/overlap checks cannot prove meaning preservation. Sentence alignments are approximate and require review. The static demonstration does not use the final model.')

elif page=='Deployment guide':
    st.title('🚀 GitHub → Streamlit Community Cloud')
    st.markdown('''1. Create a public or private GitHub repository, for example `gujarati-vishwakosh-research`.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub and select **Create app**.
4. Choose your repository, branch and main file `app.py`.
5. Deploy. The application requires no Google Drive mount, Hugging Face token, GPU, or trained model files.
6. To update the research narrative, edit the stage descriptions and documented loss data in `app.py`.''')
    st.info('Optional: add a README and screenshots. Do not upload private Drive data or copyrighted source books to a public repository without permission.')

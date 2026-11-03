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

# Editorial candidates provided by the researcher; NOT validated Vishwakosh corpus rules.
RULES = [
    ('R001', 'કહેવામાં આવે છે', 'કહેવાય છે', 'conservative', 'Definitions'),
    ('R002', 'તરીકે ઓળખવામાં આવે છે', 'તરીકે ઓળખાય છે', 'conservative', 'Definitions'),
    ('R003', 'નામથી ઓળખવામાં આવે છે', 'નામે ઓળખાય છે', 'conservative', 'Definitions'),
    ('R004', 'ઉપયોગ કરવામાં આવે છે', 'ઉપયોગ થાય છે', 'review', 'Definitions'),
    ('R005', 'વર્ણન કરવામાં આવ્યું છે', 'વર્ણવાયું છે', 'context', 'Definitions'),
    ('R006', 'રજૂ કરવામાં આવ્યું છે', 'રજૂ થયું છે', 'review', 'Definitions'),
    ('R007', 'દર્શાવવામાં આવ્યું છે', 'દર્શાવાયું છે', 'context', 'Definitions'),
    ('R008', 'જોવા મળતું હોય છે', 'જોવા મળે છે', 'conservative', 'Definitions'),
    ('R009', 'માનવામાં આવે છે', 'મનાય છે', 'conservative', 'Definitions'),
    ('R010', 'ગણવામાં આવે છે', 'ગણાય છે', 'conservative', 'Definitions'),
    ('R011', 'સમજાવવામાં આવ્યું છે', 'સમજાવાયું છે', 'context', 'Definitions'),
    ('R012', 'સમાવેશ કરવામાં આવે છે', 'સમાવેશ થાય છે', 'review', 'Definitions'),
    ('R013', 'રચના કરવામાં આવી હતી', 'રચના થઈ હતી', 'review', 'Definitions'),
    ('R014', 'સ્થાપના કરવામાં આવી હતી', 'સ્થાપના થઈ હતી', 'review', 'Definitions'),
    ('R015', 'શોધ કરવામાં આવી હતી', 'શોધ થઈ હતી', 'review', 'Definitions'),
    ('R016', 'વિકાસ કરવામાં આવ્યો હતો', 'વિકાસ થયો હતો', 'review', 'Definitions'),
    ('R017', 'પ્રયોગ કરવામાં આવ્યો હતો', 'પ્રયોગ થયો હતો', 'review', 'Definitions'),
    ('R018', 'પરિવર્તન કરવામાં આવ્યું હતું', 'પરિવર્તન થયું હતું', 'review', 'Definitions'),
    ('R019', 'ઉલ્લેખ કરવામાં આવ્યો છે', 'ઉલ્લેખ થયો છે', 'review', 'Definitions'),
    ('R020', 'અભ્યાસ કરવામાં આવે છે', 'અભ્યાસ થાય છે', 'review', 'Definitions'),
    ('R021', 'આ બાબત વિશે', 'આ અંગે', 'conservative', 'Vocabulary'),
    ('R022', 'આ વિષય વિશે', 'આ વિષય અંગે', 'conservative', 'Vocabulary'),
    ('R023', 'તેના વિશેની માહિતી', 'તે અંગેની માહિતી', 'conservative', 'Vocabulary'),
    ('R024', 'માહિતી મળી શકે છે', 'માહિતી પ્રાપ્ત થઈ શકે છે', 'context', 'Vocabulary'),
    ('R025', 'ખૂબ જ મહત્વનું', 'અત્યંત મહત્વનું', 'context', 'Vocabulary'),
    ('R026', 'ખૂબ મોટો', 'વિશાળ', 'review', 'Vocabulary'),
    ('R027', 'ખૂબ નાનો', 'અતિસૂક્ષ્મ', 'review', 'Vocabulary'),
    ('R028', 'અલગ અલગ પ્રકારના', 'વિવિધ પ્રકારના', 'conservative', 'Vocabulary'),
    ('R029', 'જુદા જુદા પ્રકારના', 'વિવિધ પ્રકારના', 'conservative', 'Vocabulary'),
    ('R030', 'ઘણા બધા', 'અનેક', 'review', 'Vocabulary'),
    ('R031', 'ઘણાં પ્રકારનાં', 'વિવિધ', 'review', 'Vocabulary'),
    ('R032', 'મુખ્ય કારણ', 'પ્રમુખ કારણ', 'review', 'Vocabulary'),
    ('R033', 'મહત્વની ભૂમિકા', 'નોંધપાત્ર ભૂમિકા', 'review', 'Vocabulary'),
    ('R034', 'ખાસ કરીને', 'વિશેષ કરીને', 'conservative', 'Vocabulary'),
    ('R035', 'આ ઉપરાંત', 'ઉપરાંત', 'conservative', 'Vocabulary'),
    ('R036', 'તેના કારણે', 'પરિણામે', 'review', 'Vocabulary'),
    ('R037', 'આ રીતે', 'આ પ્રમાણે', 'review', 'Vocabulary'),
    ('R038', 'આ માટે', 'આ હેતુથી', 'review', 'Vocabulary'),
    ('R039', 'ફરી ઉપલબ્ધ થતા', 'પુનઃ ઉપલબ્ધ થતા', 'conservative', 'Vocabulary'),
    ('R040', 'ઊંચા તાપમાને', 'ઉચ્ચ તાપમાને', 'conservative', 'Vocabulary'),
    ('R041', 'અલગ ચિહ્નો', 'ભિન્ન ચિહ્નો', 'conservative', 'Technical'),
    ('R042', 'જુદી પદ્ધતિ', 'ભિન્ન પદ્ધતિ', 'conservative', 'Technical'),
    ('R043', 'અલગ સ્વરૂપ', 'ભિન્ન સ્વરૂપ', 'conservative', 'Technical'),
    ('R044', 'તપાસ કરવામાં આવે છે', 'પરીક્ષણ કરવામાં આવે છે', 'review', 'Technical'),
    ('R045', 'માપવામાં આવે છે', 'માપન થાય છે', 'review', 'Technical'),
    ('R046', 'ગણતરી કરવામાં આવે છે', 'ગણતરી થાય છે', 'review', 'Technical'),
    ('R047', 'ઉપયોગમાં લેવામાં આવે છે', 'ઉપયોગમાં લેવાય છે', 'conservative', 'Technical'),
    ('R048', 'નોંધવામાં આવે છે', 'નોંધાય છે', 'conservative', 'Technical'),
    ('R049', 'સંગ્રહ કરવામાં આવે છે', 'સંગ્રહાય છે', 'review', 'Technical'),
    ('R050', 'ઉત્પન્ન કરવામાં આવે છે', 'ઉત્પન્ન થાય છે', 'review', 'Technical'),
    ('R051', 'બનાવવામાં આવે છે', 'બને છે', 'review', 'Technical'),
    ('R052', 'મેળવવામાં આવે છે', 'પ્રાપ્ત થાય છે', 'review', 'Technical'),
    ('R053', 'મોકલવામાં આવે છે', 'મોકલાય છે', 'context', 'Technical'),
    ('R054', 'દર્શાવવામાં આવે છે', 'દર્શાવાય છે', 'conservative', 'Technical'),
    ('R055', 'ઓળખવામાં આવે છે', 'ઓળખાય છે', 'conservative', 'Technical'),
    ('R056', 'વહેંચવામાં આવે છે', 'વહેંચાય છે', 'review', 'Technical'),
    ('R057', 'બદલવામાં આવે છે', 'બદલાય છે', 'review', 'Technical'),
    ('R058', 'વધારવામાં આવે છે', 'વધે છે', 'review', 'Technical'),
    ('R059', 'ઘટાડવામાં આવે છે', 'ઘટે છે', 'review', 'Technical'),
    ('R060', 'નિયંત્રિત કરવામાં આવે છે', 'નિયંત્રિત થાય છે', 'review', 'Technical'),
    ('R061', 'જન્મ થયો હતો', 'જન્મ થયો', 'context', 'Historical'),
    ('R062', 'મૃત્યુ થયું હતું', 'અવસાન થયું', 'review', 'Historical'),
    ('R063', 'જીવનકાળ દરમિયાન', 'જીવનકાળમાં', 'conservative', 'Historical'),
    ('R064', 'તે સમય દરમિયાન', 'તે સમયે', 'conservative', 'Historical'),
    ('R065', 'આ સમયગાળા દરમિયાન', 'આ સમયગાળામાં', 'conservative', 'Historical'),
    ('R066', 'પ્રાચીન સમયથી', 'પ્રાચીનકાળથી', 'conservative', 'Historical'),
    ('R067', 'જુના સમયમાં', 'પ્રાચીનકાળમાં', 'review', 'Historical'),
    ('R068', 'ઇતિહાસમાં જોવા મળે છે', 'ઇતિહાસમાં નોંધાય છે', 'review', 'Historical'),
    ('R069', 'ઉલ્લેખ જોવા મળે છે', 'ઉલ્લેખ મળે છે', 'conservative', 'Historical'),
    ('R070', 'લખવામાં આવ્યું હતું', 'લખાયું હતું', 'conservative', 'Historical'),
    ('R071', 'પ્રકાશિત કરવામાં આવ્યું હતું', 'પ્રકાશિત થયું હતું', 'review', 'Historical'),
    ('R072', 'અનુવાદ કરવામાં આવ્યો હતો', 'અનુવાદ થયો હતો', 'review', 'Historical'),
    ('R073', 'રચના કરવામાં આવી હતી', 'રચના થઈ હતી', 'review', 'Historical'),
    ('R074', 'લોકપ્રિય બન્યું હતું', 'લોકપ્રિય બન્યું', 'conservative', 'Historical'),
    ('R075', 'પ્રસિદ્ધિ મેળવી હતી', 'પ્રસિદ્ધિ મેળવી', 'conservative', 'Historical'),
    ('R076', 'જાણીતા હતા', 'પ્રસિદ્ધ હતા', 'review', 'Historical'),
    ('R077', 'મહત્વપૂર્ણ યોગદાન', 'નોંધપાત્ર યોગદાન', 'review', 'Historical'),
    ('R078', 'અનેક પુસ્તકો', 'અનેક ગ્રંથો', 'review', 'Historical'),
    ('R079', 'લખાણમાં ઉલ્લેખ છે', 'લખાણમાં ઉલ્લેખ મળે છે', 'conservative', 'Historical'),
    ('R080', 'માહિતી આપવામાં આવી છે', 'માહિતી અપાઈ છે', 'conservative', 'Historical'),
    ('R081', 'એ વાત નોંધવી જરૂરી છે કે', 'નોંધનીય છે કે', 'review', 'Conciseness'),
    ('R082', 'એ સમજવું મહત્વનું છે કે', 'નોંધનીય છે કે', 'review', 'Conciseness'),
    ('R083', 'આ બાબત ધ્યાનમાં રાખવી જોઈએ કે', 'ધ્યાનમાં રાખવું જોઈએ કે', 'conservative', 'Conciseness'),
    ('R084', 'આના પરિણામ સ્વરૂપે', 'પરિણામે', 'review', 'Conciseness'),
    ('R085', 'આ કારણને લીધે', 'આ કારણે', 'conservative', 'Conciseness'),
    ('R086', 'તેના પરિણામ તરીકે', 'પરિણામે', 'review', 'Conciseness'),
    ('R087', 'આ હેતુ માટે', 'આ માટે', 'conservative', 'Conciseness'),
    ('R088', 'તેના ઉપયોગ દ્વારા', 'તેના ઉપયોગથી', 'conservative', 'Conciseness'),
    ('R089', 'આ પ્રક્રિયા દરમિયાન', 'આ પ્રક્રિયામાં', 'conservative', 'Conciseness'),
    ('R090', 'વિવિધ પ્રકારની પદ્ધતિઓ', 'વિવિધ પદ્ધતિઓ', 'conservative', 'Conciseness'),
    ('R091', 'ઘણા જુદા જુદા પ્રકારના', 'અનેક પ્રકારના', 'conservative', 'Conciseness'),
    ('R092', 'સંપૂર્ણ રીતે', 'સંપૂર્ણપણે', 'conservative', 'Conciseness'),
    ('R093', 'સામાન્ય રીતે જોવા મળે છે', 'સામાન્ય રીતે મળે છે', 'review', 'Conciseness'),
    ('R094', 'તે જોવા મળ્યું હતું કે', 'એવું જોવા મળ્યું કે', 'review', 'Conciseness'),
    ('R095', 'એવો અંદાજ લગાવવામાં આવે છે કે', 'એવો અંદાજ છે કે', 'review', 'Conciseness'),
    ('R096', 'એવું માનવામાં આવે છે કે', 'એવું મનાય છે કે', 'conservative', 'Conciseness'),
    ('R097', 'તેવું કહી શકાય છે કે', 'કહી શકાય કે', 'conservative', 'Conciseness'),
    ('R098', 'આ રીતે કહી શકાય કે', 'કહી શકાય કે', 'conservative', 'Conciseness'),
    ('R099', 'નીચે મુજબ દર્શાવવામાં આવ્યું છે', 'નીચે દર્શાવ્યું છે', 'review', 'Conciseness'),
    ('R100', 'ઉપર જણાવ્યા પ્રમાણે', 'ઉપર જણાવ્યા મુજબ', 'conservative', 'Conciseness'),
]
assert len(RULES) == 100

# Default to conservative replacements. Context-sensitive candidates require opt-in.
def edit_with_audit(source, remove_citations=True, normalize_spacing=True, risk_level='conservative', enabled_ids=None):
    base, formatting = static_edit(source, remove_citations, normalize_spacing, False, None)
    result = base
    changes = [{'id':'FORMAT', 'before':'', 'after':'', 'count':1, 'risk':'format', 'note':x} for x in formatting]
    allowed = {'conservative'} if risk_level=='conservative' else {'conservative','context'} if risk_level=='context' else {'conservative','context','review'}
    enabled_ids = enabled_ids if enabled_ids is not None else {r[0] for r in RULES}
    # Longer matches first, preventing shorter matches from consuming more specific patterns.
    for rid, old, new, risk, category in sorted(RULES, key=lambda r:(-len(r[1]),r[0])):
        if rid not in enabled_ids or risk not in allowed or old == new:
            continue
        count = result.count(old)
        if count:
            result = result.replace(old,new)
            changes.append({'id':rid,'before':old,'after':new,'count':count,'risk':risk,'note':category})
    # Preserve numerals, and reject unexpected disappearance of non-citation numbers.
    if Counter(NUMBERS.findall(source)) != Counter(NUMBERS.findall(result)):
        return base, [x for x in changes if x['id']=='FORMAT'], 'Number mismatch: reverted phrase substitutions'
    return result, changes, ''

def text_metrics(t):
    s=[x.strip() for x in re.split(r'(?<=[.!?।])\s+',t) if x.strip()]
    words=t.split()
    return {'Characters':len(t),'Words':len(words),'Sentences':len(s),'Mean words / sentence':round(sum(len(x.split()) for x in s)/len(s),2) if s else 0,'Citation markers':len(CITATION.findall(t)),'Repeated spaces':len(re.findall(r' {2,}',t)),'Paragraphs':len([x for x in re.split(r'\n\s*\n',t) if x.strip()])}

def pipeline(items):
    for i,(title,detail) in enumerate(items,1):
        st.markdown(f'<div class="step"><b>{i:02d} · {title}</b><div class="small">{detail}</div></div>',unsafe_allow_html=True)
        if i<len(items): st.markdown('<div style="margin-left:24px;color:#4975c7;font-size:1.2rem">↓</div>',unsafe_allow_html=True)

st.sidebar.markdown('## 📚 Gujarati Vishwakosh')
st.sidebar.caption('Research and editorial workspace')
page=st.sidebar.radio('Explore', ['Overview','Gujarati editor','Research journey','Final ByT5 pipeline','Training analytics','Model comparison','Style feature matrix','Evaluation & limitations','Deployment guide'])
st.sidebar.divider()
st.sidebar.caption('Editorial rules with transparent change tracking. No model inference in this app.')

if page=='Overview':
    st.markdown('<div class="hero"><h1>Gujarati Vishwakosh · Style-Controlled SLM Research</h1><p>Gujarati encyclopedic editing • multilingual small language models • transparent, fact-preserving workflows</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('Source articles',44);b.metric('Converted references',43);c.metric('Candidate training pairs',1461);d.metric('Final validation loss', '0.4335')
    st.subheader('Research objective')
    st.write('Study Gujarati Vishwakosh editorial conventions and develop an editor that transforms existing Wikipedia-style Gujarati into formal encyclopedic Gujarati while preserving facts, dates, names, and meaning. The final reported trained-model experiment uses **google/byt5-small**. This public demonstration intentionally uses **static rules only**.')
    st.markdown('<span class="chip">Gujarati NLP</span><span class="chip">Text-to-text editing</span><span class="chip">ByT5</span><span class="chip">mT5</span><span class="chip">Evidence-based evaluation</span>',unsafe_allow_html=True)
    st.subheader('Research flow at a glance')
    pipeline([('Corpus preparation','Wikipedia articles + converted Vishwakosh-style references + Excel reference'),('Style analysis','Editorial features, measurable diagnostics, protected facts'),('Model exploration','Rule-based teacher, mT5 experiments, ByT5 experiments'),('Final trained experiment','ByT5-small; manual PyTorch training, 5 epochs'),('Evaluation','Held-out loss, generation checks, 20 Gujarati tests; human fidelity review still required'),('Presentation-only deployment','Interactive editorial editor with no model weights or GPU')])

elif page=='Gujarati editor':
    st.markdown('<div class="hero"><h1>ગુજરાતી વિશ્વકોશ · સંપાદક</h1><p>Gujarati editorial refinement · inspect every change · preserve source facts</p></div>',unsafe_allow_html=True)
    st.caption('Rules are editorial candidates, not independently verified Vishwakosh usage. Context-dependent edits need review.')
    src=st.text_area('મૂળ ગુજરાતી લખાણ',value=EXAMPLE,height=210,key='editor_input')
    c1,c2,c3=st.columns([1,1,2])
    citation=c1.checkbox('Remove citations',True)
    spacing=c2.checkbox('Clean punctuation',True)
    level=c3.selectbox('Editorial mode',['Conservative','Context review','All candidates (review required)'],index=0)
    risk_level={'Conservative':'conservative','Context review':'context','All candidates (review required)':'all'}[level]
    with st.expander('Editorial rules · select and inspect',expanded=False):
        groups=sorted(set(r[4] for r in RULES))
        group=st.selectbox('Category',['All']+groups)
        query=st.text_input('Search rule ID or Gujarati phrase')
        display=[r for r in RULES if (group=='All' or r[4]==group) and (not query or query.lower() in (' '.join(r)).lower())]
        st.dataframe(pd.DataFrame(display,columns=['ID','Original','Replacement','Review level','Category']),hide_index=True,use_container_width=True)
        st.caption('Conservative rules run by default. Other candidates are opt-in and may change agency, emphasis or meaning.')
        st.download_button('Export 100-rule catalog',pd.DataFrame(RULES,columns=['ID','Original','Replacement','Review level','Category']).to_csv(index=False).encode('utf-8-sig'),file_name='gujarati_editorial_rules.csv',mime='text/csv')
    if 'conversion_result' not in st.session_state:
        st.session_state.conversion_result = None
    if st.button('✍️ Convert to Vishwakosh Gujarati · રૂપાંતર કરો', type='primary', use_container_width=True):
        if not src.strip():
            st.warning('કૃપા કરીને ગુજરાતી લખાણ દાખલ કરો.')
        else:
            edited_now, changes_now, warning_now = edit_with_audit(src, citation, spacing, risk_level)
            st.session_state.conversion_result = (src, edited_now, changes_now, warning_now)
    if st.session_state.conversion_result is None:
        st.info('ઉપર તમારો ગુજરાતી ફકરો લખો અને **રૂપાંતર કરો** બટન દબાવો.')
        st.stop()
    original, edited, changes, warning = st.session_state.conversion_result
    if original != src:
        st.info('Input has changed since the last conversion. Click **રૂપાંતર કરો** to update the result.')
    left,right=st.columns(2)
    with left:
        st.markdown('#### Original')
        st.text_area('Original paragraph',original,height=250,disabled=True,label_visibility='collapsed',key='original_panel')
    with right:
        st.markdown('#### Edited')
        st.text_area('Edited paragraph',edited,height=250,label_visibility='collapsed',key='edited_panel')
    a,b,c=st.columns(3)
    a.metric('Applied operations',sum(x['count'] for x in changes))
    b.metric('Rules triggered',len({x['id'] for x in changes if x['id']!='FORMAT'}))
    c.metric('Words',len(edited.split()))
    if warning: st.error(warning)
    if changes:
        st.markdown('#### Applied changes')
        st.dataframe(pd.DataFrame(changes),hide_index=True,use_container_width=True)
    else: st.info('No applicable edits found for this paragraph. The text was preserved.')
    if any(x['risk']!='conservative' and x['risk']!='format' for x in changes):
        st.warning('Context-sensitive substitutions applied. Compare with the source before using the output.')
    st.download_button('Download edited text',edited.encode('utf-8'),file_name='gujarati_vishwakosh_edit.txt',mime='text/plain')
    with st.expander('Before / after measurements'):
        st.dataframe(pd.DataFrame({'Before':text_metrics(original),'After':text_metrics(edited)}),use_container_width=True)
        st.caption('Number retention and word counts cannot prove semantic fidelity. Human checking is needed.')

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
      ('Stage G · Streamlit showcase','No weights, no training, no inference','Deterministic editorial cleanup and interactive PhD research presentation.','Live application demonstration')]
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
    st.graphviz_chart('''digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fillcolor="#eaf2ff" color="#4475b8" fontname="Helvetica" fontsize=11]; A[label="Gujarati source"]; B[label="ByT5 encoder"]; C[label="ByT5 decoder"]; D[label="Raw rewrite"]; E[label="Safety audit" fillcolor="#fff0d6"]; F[label="Accepted output" fillcolor="#dff5e8"]; G[label="Guarded fallback" fillcolor="#f8e5e5"]; A->B->C->D->E; E->F[label="pass"]; E->G[label="reject"]; }''',use_container_width=True)
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

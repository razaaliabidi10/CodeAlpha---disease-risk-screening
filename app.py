  # Streamlit app: disease risk screening (v4, text inputs)
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

PROJECT_DIR = str(Path(__file__).parent)
GREEN, RISK, GREY, INK = '#1E5B45', '#B5472F', '#D5DDD7', '#1C2420'

st.set_page_config(page_title='Disease Risk Screening', layout='wide', initial_sidebar_state='collapsed')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp input {font-family: 'Inter', sans-serif;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stHeader"], footer {display: none;}
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.st-key-card_left, .st-key-card_right {background: #FFFFFF; border: 1px solid #C5D0C8 !important; border-radius: 16px; box-shadow: 0 2px 12px rgba(30,91,69,0.10); padding: 0.8rem 1rem;}
.stButton button {width: 100%; border-radius: 10px;}
[data-testid="stFormSubmitButton"] button {width: 100%; background: #1E5B45 !important; color: #FFFFFF !important; border: 0; border-radius: 10px; padding: 0.7rem 1rem;}
.nav {background: #1E5B45; border-radius: 14px; padding: 0.9rem 1.3rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;}
.brand {font-weight: 700; font-size: 1.2rem; color: #FFFFFF;}
.navtag {color: #CFE3D8; font-size: 0.85rem;}
.verdict {font-size: 1.8rem; font-weight: 700; line-height: 1.2;}
.score {font-size: 1.05rem; color: #3D4640; margin-top: 0.3rem;}
.empty {text-align: center; padding: 3rem 1rem; color: #6B746E;}
</style>""", unsafe_allow_html=True)

NAMES = {'heart': 'Heart disease', 'diabetes': 'Diabetes', 'cancer': 'Breast cancer'}
DISEASE = {'heart': 'heart disease', 'diabetes': 'diabetes', 'cancer': 'breast cancer'}
POP = {
    'heart': 'Trained on 303 patients from the Cleveland clinic (UCI), mostly older adults referred for suspected heart problems.',
    'diabetes': 'Trained on 768 women of Pima Indian heritage aged 21+. Not valid for men or other populations.',
    'cancer': 'Trained on 569 fine-needle-aspirate samples (Wisconsin). Inputs are lab measurements of cell nuclei.'
}
SPECS = {
    'heart': {
        'age': ('num', 'Age (years)'),
        'sex': ('cat', 'Sex', {0: 'Female', 1: 'Male'}),
        'cp': ('cat', 'Chest pain type', {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'}),
        'trestbps': ('num', 'Resting blood pressure (mm Hg)'),
        'chol': ('num', 'Cholesterol (mg/dl)'),
        'fbs': ('cat', 'Fasting blood sugar above 120 mg/dl', {0: 'No', 1: 'Yes'}),
        'restecg': ('cat', 'Resting ECG', {0: 'Normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'}),
        'thalach': ('num', 'Maximum heart rate (bpm)'),
        'exang': ('cat', 'Exercise-induced angina', {0: 'No', 1: 'Yes'}),
        'oldpeak': ('num', 'ST depression (oldpeak)'),
        'slope': ('cat', 'ST slope', {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}),
        'ca': ('cat', 'Major vessels (fluoroscopy)', {0: '0', 1: '1', 2: '2', 3: '3'}),
        'thal': ('cat', 'Thalassemia', {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'})
    },
    'diabetes': {
        'pregnancies': ('num', 'Pregnancies'),
        'glucose': ('num', 'Plasma glucose (mg/dl)'),
        'blood_pressure': ('num', 'Diastolic blood pressure (mm Hg)'),
        'skin_thickness': ('num', 'Skin fold thickness (mm)'),
        'insulin': ('num', 'Serum insulin (mu U/ml)'),
        'bmi': ('num', 'BMI'),
        'diabetes_pedigree': ('num', 'Diabetes pedigree function'),
        'age': ('num', 'Age (years)')
    }
}
PRIMARY = {
    'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'exang'],
    'diabetes': ['age', 'glucose', 'bmi', 'blood_pressure', 'pregnancies']
}
REQUIRED = {'heart': ['age', 'sex'], 'diabetes': ['glucose'], 'cancer': []}
BIN = {'heart': ['sex', 'fbs', 'exang']}
PH = {'age': '55', 'trestbps': '140', 'chol': '240', 'thalach': '150', 'oldpeak': '1.5', 'glucose': '110', 'bmi': '28',
      'blood_pressure': '70', 'pregnancies': '2', 'insulin': '80', 'skin_thickness': '25', 'diabetes_pedigree': '0.5'}


@st.cache_resource
def load_all():
    return joblib.load(f'{PROJECT_DIR}/models/app_bundle.pkl'), joblib.load(f'{PROJECT_DIR}/data/splits.pkl')


@st.cache_data
def load_tables():
    return {k: pd.read_csv(f'{PROJECT_DIR}/outputs/{k}.csv') for k in ['cv_results', 'tuned_results', 'final_test_results', 'final_test_all_models']}


BUNDLE, SPLITS = load_all()
T = load_tables()


def spec(name, f):
    return SPECS.get(name, {}).get(f, ('num', f.capitalize()))


def short(label):
    return label.split(' (')[0]


def wkey(name, f):
    return f'{name}_f4_{f}'


def sig(values):
    return tuple((f, None if np.isnan(v) else v) for f, v in values.items())


def prepare(name, raw):
    v = dict(raw)
    for f in BIN.get(name, []):
        if np.isnan(v.get(f, np.nan)):
            v[f] = BUNDLE[name]['stats'][f]['median']
    return v


def predict(name, values):
    X = pd.DataFrame([prepare(name, values)], columns=BUNDLE[name]['features']).astype(float)
    return float(BUNDLE[name]['model'].predict_proba(X)[0, 1])


def effects(name, values, given):
    base = predict(name, values)
    return pd.Series({f: base - predict(name, {**values, f: np.nan}) for f in given})


def describe(name, f, v):
    kind, label, *rest = spec(name, f)
    if kind == 'cat':
        return f'{short(label)}: {rest[0].get(int(v), int(v))}'
    return f'{short(label)}: {v:g}'


def reset(name):
    for k in [k for k in st.session_state if k.startswith(f'{name}_')]:
        del st.session_state[k]


def fill(name, vals):
    for f, v in vals.items():
        if not np.isnan(v):
            st.session_state[wkey(name, f)] = int(v) if spec(name, f)[0] == 'cat' else f'{v:g}'


def load_patient(name):
    reset(name)
    X_te, y_te = SPLITS[name][1], SPLITS[name][3]
    i = np.random.randint(len(X_te))
    row = X_te.iloc[i]
    vals = {f: (np.nan if pd.isna(row[f]) else float(row[f])) for f in BUNDLE[name]['features']}
    fill(name, vals)
    given = [f for f, v in vals.items() if not np.isnan(v)]
    st.session_state[f'{name}_result'] = {'values': vals, 'given': given, 'truth': int(y_te.iloc[i])}



def field(name, f, col):
    kind, label, *rest = spec(name, f)
    if kind == 'cat':
        opts = rest[0]
        return col.selectbox(label, list(opts), index=None, format_func=opts.get, placeholder='Select', key=wkey(name, f))
    ph = 'e.g. ' + PH.get(f, f"{BUNDLE[name]['stats'][f]['median']:.4g}")
    return col.text_input(label, placeholder=ph, key=wkey(name, f))


def risk_bar(p, t):
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, x1=1, y0=0, y1=1, fillcolor=GREY, line_width=0)
    fig.add_shape(type='rect', x0=0, x1=p, y0=0, y1=1, fillcolor=RISK if p >= t else GREEN, line_width=0)
    fig.add_shape(type='line', x0=t, x1=t, y0=-0.25, y1=1.25, line=dict(color=INK, width=2))
    fig.add_annotation(x=t, y=1.55, text=f'flag threshold {t:.0%}', showarrow=False, font=dict(size=12, color=INK))
    fig.update_xaxes(range=[0, 1], tickformat='.0%', showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[-0.4, 1.9])
    fig.update_layout(height=110, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def tradeoff(name, t):
    y = SPLITS[name][3].values
    p = BUNDLE[name]['model'].predict_proba(SPLITS[name][1])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    rec = [((p >= g) & (y == 1)).sum() / (y == 1).sum() for g in grid]
    fa = [((p >= g) & (y == 0)).sum() / (y == 0).sum() for g in grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=rec, name='Sick patients caught', line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=grid, y=fa, name='Healthy patients falsely flagged', line=dict(color=RISK, width=3)))
    fig.add_vline(x=t, line_color=INK, line_width=2)
    fig.update_yaxes(tickformat='.0%', range=[0, 1.02], gridcolor=GREY)
    fig.update_xaxes(title='Flag threshold', gridcolor=GREY)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=-0.3))
    return fig, int(((p >= t) & (y == 1)).sum()), int((y == 1).sum()), int(((p >= t) & (y == 0)).sum()), int((y == 0).sum())


st.markdown("<div class='nav'><span class='brand'>Disease Risk Screening</span><span class='navtag'>A machine-learning research project</span></div>", unsafe_allow_html=True)
name = st.radio('Condition', list(NAMES), format_func=NAMES.get, horizontal=True, label_visibility='collapsed')
feats = BUNDLE[name]['features']
tuned_t = BUNDLE[name]['threshold']
t = st.session_state.get(f'thr_{name}', round(tuned_t, 2))
_r = st.session_state.get(f'{name}_result')
if _r and not any(wkey(name, f) in st.session_state for f in feats):
    fill(name, _r['values'])
st.markdown(f'## {NAMES[name]} prediction')
st.caption(POP[name])
tab1, tab2, tab3 = st.tabs(['Prediction', 'Model comparison', 'About'])

with tab1:
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        with st.container(border=True, key='card_left'):
            st.markdown('#### Your measurements')
            req = ' and '.join(short(spec(name, f)[1]).lower() for f in REQUIRED[name])
            st.caption(f'Start with {req}. Add the details you have.' if req else 'Fill in what you have.')
            b1, b2 = st.columns(2)
            b1.button('Load sample patient', on_click=load_patient, args=(name,), key=f'sample_{name}')
            b2.button('Reset', on_click=reset, args=(name,), key=f'reset_{name}')
            primary = PRIMARY.get(name, [f for f in feats if f.startswith('mean')])
            more = [f for f in feats if f not in primary]
            raw = {}
            with st.form(f'form_{name}', border=False):
                cols = st.columns(2)
                for i, f in enumerate(primary):
                    raw[f] = field(name, f, cols[i % 2])
                if more:
                    with st.expander('More details (optional)'):
                        cols = st.columns(2)
                        for i, f in enumerate(more):
                            raw[f] = field(name, f, cols[i % 2])
                go_btn = st.form_submit_button('Get prediction', type='primary')
            st.caption('Type your values, then click Get prediction. Blank fields use typical values from the training data. Nothing is saved.')
            if go_btn:
                vals, bad = {}, []
                for f in feats:
                    v = raw.get(f)
                    if v is None or str(v).strip() == '':
                        vals[f] = np.nan
                        continue
                    try:
                        x = float(str(v).strip().replace(',', '.'))
                    except ValueError:
                        x = -1.0
                    if x < 0 or x > 3 * BUNDLE[name]['stats'][f]['max']:
                        bad.append(short(spec(name, f)[1]))
                        vals[f] = np.nan
                    else:
                        vals[f] = x
                given = [f for f, v in vals.items() if not np.isnan(v)]
                miss = [short(spec(name, f)[1]) for f in REQUIRED[name] if f not in given]
                if bad:
                    st.error('These values do not look right (numbers only, no negatives): ' + ', '.join(bad))
                elif miss:
                    st.error('Please enter: ' + ', '.join(miss))
                elif not given:
                    st.error('Enter at least one measurement.')
                else:
                    prev = st.session_state.get(f'{name}_result')
                    keep = prev is not None and sig(prev['values']) == sig(vals)
                    st.session_state[f'{name}_result'] = {'values': vals, 'given': given, 'truth': prev['truth'] if keep else None}

    with right:
        with st.container(border=True, key='card_right'):
            st.markdown('#### Prediction result')
            res = st.session_state.get(f'{name}_result')
            if res is None:
                st.markdown("<div class='empty'><div style='font-size:2rem'>&#8767;</div>Your prediction will appear here.<br>Enter measurements, then select Get prediction.</div>", unsafe_allow_html=True)
            else:
                values, given = res['values'], res['given']
                p = predict(name, values)
                flagged = p >= t
                c = RISK if flagged else GREEN
                head = f'Higher risk of {DISEASE[name]}' if flagged else f'Lower risk of {DISEASE[name]}'
                sub = 'The model flags this profile. A follow-up test or doctor visit is recommended.' if flagged else 'The model does not flag this profile. This does not rule anything out.'
                st.markdown(f"<div style='background:{c}14;border-left:6px solid {c};border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.6rem'>"
                            f"<div class='verdict' style='color:{c}'>{head}</div><div class='score'>Model score {p:.0%}</div></div>", unsafe_allow_html=True)
                st.write(sub)
                st.plotly_chart(risk_bar(p, t), width='stretch')
                st.caption(f'Based on {len(given)} of {len(feats)} inputs. The score is not a true probability.')
                if len(given) < 0.3 * len(feats):
                    st.warning('Few inputs given. The rest use typical values, so treat this score as weak.')
                out = [short(spec(name, f)[1]) for f in given if not BUNDLE[name]['stats'][f]['min'] <= values[f] <= BUNDLE[name]['stats'][f]['max']]
                if out:
                    st.warning('Outside the range seen in training: ' + ', '.join(out) + '. Interpret with caution.')
                if res['truth'] is not None:
                    st.info('Real outcome of this sample patient: ' + ('sick' if res['truth'] else 'not sick') + '.')
                with st.expander('Explain result'):
                    eff = effects(name, values, given)
                    top = eff.reindex(eff.abs().sort_values(ascending=False).head(6).index).iloc[::-1]
                    fig = go.Figure(go.Bar(x=top.values * 100, y=[describe(name, f, values[f]) for f in top.index], orientation='h',
                                           marker_color=[RISK if v > 0 else GREEN for v in top.values]))
                    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_title='score points vs leaving this blank', showlegend=False)
                    fig.update_xaxes(zeroline=True, zerolinecolor=INK, gridcolor=GREY)
                    st.plotly_chart(fig, width='stretch')
                    st.caption('Red raises the score, green lowers it, compared with a typical patient.')
                    if name == 'cancer':
                        st.caption('Cancer measurements are highly correlated, so individual effects are unstable. Do not read them as causes.')
            with st.expander('Adjust sensitivity (advanced)'):
                st.slider('Flag threshold', 0.05, 0.95, round(tuned_t, 2), 0.01, key=f'thr_{name}')
                st.caption(f'Tuned value: {tuned_t:.2f}. Lower catches more sick patients but raises false alarms.')

with tab2:
    st.subheader('Sensitivity trade-off on held-out test patients')
    fig, caught, n_sick, fl, n_h = tradeoff(name, t)
    st.plotly_chart(fig, width='stretch')
    st.markdown(f'At threshold **{t:.2f}**: catches **{caught} of {n_sick}** sick patients and falsely flags **{fl} of {n_h}** healthy patients.')
    st.caption('The test set is small, so every number here has a margin of error of several points.')
    st.subheader('Algorithm comparison (5-fold cross-validation)')
    metric = st.selectbox('Compare by', ['roc_auc', 'recall', 'precision', 'f1', 'accuracy'], key=f'metric_{name}')
    cv = T['cv_results']
    cv = cv[cv.dataset == name]
    show = pd.DataFrame({'Model': [m + (' (selected)' if m == 'Logistic Regression' else '') for m in cv.model], 'Score': cv[metric].values})
    st.dataframe(show, hide_index=True, column_config={'Score': st.column_config.ProgressColumn('Score', min_value=0.0, max_value=1.0, format='%.3f')})
    st.caption('Differences smaller than about 0.05 are noise on datasets this small.')
    st.subheader('Effect of tuning (ROC-AUC)')
    tr = T['tuned_results']
    tr = tr[tr.dataset == name]
    st.dataframe(pd.DataFrame({'Model': tr.model.values, 'Before': tr.untuned_auc.values, 'After': tr.tuned_auc.values,
                               'Change': [f'{(b - a) * 100:+.1f} pp' for a, b in zip(tr.untuned_auc, tr.tuned_auc)]}), hide_index=True)
    st.subheader('Final test: all four tuned models at the default 0.5 cutoff')
    ta = T['final_test_all_models']
    st.dataframe(ta[ta.dataset == name].drop(columns='dataset'), hide_index=True)
    st.caption('Recall differences here mostly reflect how each model sets its probabilities, not better separation of patients.')
    st.subheader('Final test: selected model at both cutoffs')
    tf = T['final_test_results']
    st.dataframe(tf[tf.dataset == name].drop(columns='dataset'), hide_index=True)

with tab3:
    st.markdown("""
**Data.** UCI Heart Disease (Cleveland, 303 patients), Pima Indians Diabetes (768 women), Breast Cancer Wisconsin Diagnostic (569 samples).

**Method.** Missing values are imputed inside a pipeline fitted on training data only. Four algorithms (SVM, Logistic Regression, Random Forest, XGBoost) were compared with 5-fold cross-validation and tuned with grid search. Their scores were within noise, so the simplest and most explainable model, Logistic Regression, was chosen before the test set was opened. The flag threshold was tuned on out-of-fold predictions to reach a recall target.

**Limitations.**
Small datasets, so test results carry a wide margin of error.
Each model reflects its source population only.
Model scores are not calibrated probabilities.
Cancer: 2 of 42 malignant test samples were missed.
Educational project. Not a diagnostic tool, not clinically validated, and not medical advice.
""")

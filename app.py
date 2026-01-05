# Comments in English only
import os
import time

import streamlit as st

from src.ct_model import CTSatModel
from src.plots import make_currents_figure, make_flux_excitation_figure
from src.validation import validate_inputs, build_warnings
from src.presets import PRESETS, preset_names
from src.export import results_to_csv_bytes, results_to_json_bytes

st.set_page_config(
    page_title="CT Saturation Simulator (PSRC)",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ CT Saturation Simulator (PSRC)")
st.caption("Modelo PSRC (EDO em λ) com gráficos Plotly e presets.")
st.markdown("""
**Referência**

IEEE Power System Relaying Committee (PSRC).  
*CT SAT Calculator*.  
Documento base do IEEE C37.110 – *IEEE Guide for the Application of Current Transformers Used for Protective Relaying Purposes*.  
Disponível em:  
https://ieeexplore.ieee.org/document/10132388
""")


DEFAULTS = {
    "f_hz": 60.0,
    "t_end": 0.25,
    "pre_fault_cycles": 1.0,
    "dt": 1.0 / 12000.0,
    "S": 22.0,
    "Vs": 400.0,
    "N": 240.0,
    "Rw": 0.0,
    "Rb": 4.0,
    "Xb": 2.0,
    "Ip": 12000.0,
    "Off": 1.0,
    "XoverR": 12.0,
    "Lamrem": 0.0,
    "integrator": "RK4 (fixed step)",
    "rtol": 1.0e-6,
    "atol": 1.0e-9,
    "rp_points": 200000
}

for k in DEFAULTS:
    if k not in st.session_state:
        st.session_state[k] = DEFAULTS[k]


# Apply staged preset BEFORE creating widgets (Streamlit rule)
if "pending_preset" in st.session_state:
    preset_name = str(st.session_state["pending_preset"])
    if preset_name in PRESETS:
        p = dict(PRESETS[preset_name]["inp"])
        for k in p:
            st.session_state[k] = p[k]
    del st.session_state["pending_preset"]


def build_inp_from_state() -> dict:
    inp = {}
    inp["f_hz"] = float(st.session_state["f_hz"])
    inp["t_end"] = float(st.session_state["t_end"])
    inp["pre_fault_cycles"] = float(st.session_state["pre_fault_cycles"])
    inp["dt"] = float(st.session_state["dt"])
    inp["S"] = float(st.session_state["S"])
    inp["Vs"] = float(st.session_state["Vs"])
    inp["N"] = float(st.session_state["N"])
    inp["Rw"] = float(st.session_state["Rw"])
    inp["Rb"] = float(st.session_state["Rb"])
    inp["Xb"] = float(st.session_state["Xb"])
    inp["Ip"] = float(st.session_state["Ip"])
    inp["Off"] = float(st.session_state["Off"])
    inp["XoverR"] = float(st.session_state["XoverR"])
    inp["Lamrem"] = float(st.session_state["Lamrem"])
    return inp


def scipy_available() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def cached_rp(S: float, n_points: int) -> float:
    return CTSatModel.compute_rp_numeric(S=float(S), n_points=int(n_points))


@st.cache_data(show_spinner=False)
def cached_derived(inp: dict, rp_points: int) -> dict:
    S = float(inp["S"])
    rp = cached_rp(S=S, n_points=int(rp_points))
    derived = CTSatModel.compute_derived(inp=inp, rp=rp)
    return derived


@st.cache_data(show_spinner=False)
def cached_simulation(inp: dict, derived: dict, integrator: str, rtol: float, atol: float) -> dict:
    model = CTSatModel(inp=inp, derived=derived)
    res = model.simulate(integrator=integrator, rtol=float(rtol), atol=float(atol))
    return res


def maybe_show_reference_figure() -> None:
    st.subheader("Figura")
    uploaded = st.file_uploader("Carregar figura (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        st.image(uploaded, use_container_width=True)
        return

    fig_path = os.path.join("assets", "figure.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
        st.caption("Arquivo: assets/figure.png")
        return

    st.info("Coloque sua imagem em **assets/figure.png** ou use o upload acima.")


tab_sim, tab_presets, tab_export, tab_help = st.tabs(["Simulação", "Presets", "Exportação", "Ajuda / Notas"])


with tab_presets:
    st.subheader("Casos prontos")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        selected = st.selectbox("Selecione um preset", options=preset_names(), index=0)
        st.markdown("**Descrição:**")
        st.write(PRESETS[selected]["description"])

    with col_b:
        if st.button("Load preset"):
            st.session_state["pending_preset"] = str(selected)
            st.rerun()

    st.info("Após carregar um preset, volte para a aba Simulação e rode.")


with tab_sim:



    st.subheader("Parâmetros")

    exp_sim = st.expander("Simulação", expanded=True)
    with exp_sim:
        st.number_input("f_hz", min_value=0.1, value=float(st.session_state["f_hz"]), step=1.0, key="f_hz")
        st.number_input("t_end (s)", min_value=0.001, value=float(st.session_state["t_end"]), step=0.01, key="t_end")
        st.number_input("pre_fault_cycles", min_value=0.0, value=float(st.session_state["pre_fault_cycles"]), step=0.5, key="pre_fault_cycles")
        st.number_input("dt (s)", min_value=1.0e-6, value=float(st.session_state["dt"]), step=1.0e-5, format="%.8f", key="dt")

    exp_exc = st.expander("Curva de excitação", expanded=True)
    with exp_exc:
        st.number_input("S", min_value=1.01, value=float(st.session_state["S"]), step=1.0, key="S")
        st.number_input("Vs (V rms @ Ie=10A)", min_value=1.0, value=float(st.session_state["Vs"]), step=10.0, key="Vs")

    exp_tc = st.expander("TC e burden", expanded=False)
    with exp_tc:
        st.number_input("N (ratio 1:N)", min_value=0.1, value=float(st.session_state["N"]), step=1.0, key="N")
        st.number_input("Rw (ohm)", min_value=0.0, value=float(st.session_state["Rw"]), step=0.1, key="Rw")
        st.number_input("Rb (ohm)", min_value=0.0, value=float(st.session_state["Rb"]), step=0.1, key="Rb")
        st.number_input("Xb (ohm @ f_hz)", min_value=0.0, value=float(st.session_state["Xb"]), step=0.1, key="Xb")

    exp_fault = st.expander("Falta", expanded=False)
    with exp_fault:
        st.number_input("Ip (A rms)", min_value=0.0, value=float(st.session_state["Ip"]), step=100.0, key="Ip")
        st.number_input("Off (pu)", min_value=-1.0, max_value=1.0, value=float(st.session_state["Off"]), step=0.05, key="Off")
        st.number_input("XoverR", min_value=0.01, value=float(st.session_state["XoverR"]), step=0.5, key="XoverR")

    exp_rem = st.expander("Remanência", expanded=False)
    with exp_rem:
        st.number_input("Lamrem (pu)", min_value=-1.0, max_value=1.0, value=float(st.session_state["Lamrem"]), step=0.05, key="Lamrem")

    exp_int = st.expander("Integrador", expanded=False)
    with exp_int:
        integ_opts = ["RK4 (fixed step)", "SciPy solve_ivp (RK45)"]
        if scipy_available() is False:
            st.warning("SciPy não encontrado. Opção SciPy fará fallback para RK4.")
        st.selectbox("Método", options=integ_opts, index=integ_opts.index(str(st.session_state["integrator"])), key="integrator")

        st.number_input("RP n_points", min_value=20000, value=int(st.session_state["rp_points"]), step=20000, key="rp_points")
        st.number_input("rtol (SciPy)", min_value=1.0e-12, value=float(st.session_state["rtol"]), step=1.0e-6, format="%.2e", key="rtol")
        st.number_input("atol (SciPy)", min_value=1.0e-15, value=float(st.session_state["atol"]), step=1.0e-9, format="%.2e", key="atol")

    st.divider()

    # Validate inputs + warnings
    inp_now = build_inp_from_state()
    errors = validate_inputs(inp=inp_now)
    if len(errors) > 0:
        for e in errors:
            st.error(e)
        st.stop()

    for w in build_warnings(inp=inp_now):
        st.warning(w)

    # Narrow run button
    
    col_run, col_img = st.columns([1, 3])

    with col_run:
        run_clicked = st.button("Run simulation", type="primary")

    with col_img:
        img_path = os.path.join("assets", "tc-circuit.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)


    if run_clicked is False and "last_result" not in st.session_state:
        st.info("Clique em **Run simulation** para calcular e plotar.")
        st.stop()

    if run_clicked is True:
        rp_points = int(st.session_state["rp_points"])
        derived = cached_derived(inp=inp_now, rp_points=rp_points)

        integ = str(st.session_state["integrator"])
        if integ == "SciPy solve_ivp (RK45)" and scipy_available() is False:
            integ_use = "RK4 (fixed step)"
        else:
            integ_use = integ

        start = time.perf_counter()
        res = cached_simulation(
            inp=inp_now,
            derived=derived,
            integrator=integ_use,
            rtol=float(st.session_state["rtol"]),
            atol=float(st.session_state["atol"])
        )
        elapsed = time.perf_counter() - start

        st.session_state["last_inp"] = inp_now
        st.session_state["last_derived"] = derived
        st.session_state["last_result"] = res
        st.session_state["last_elapsed"] = float(elapsed)

    last_inp = st.session_state["last_inp"]
    last_derived = st.session_state["last_derived"]
    last_result = st.session_state["last_result"]
    last_elapsed = float(st.session_state.get("last_elapsed", 0.0))

    st.caption(f"Execução: {last_elapsed:.4f} s")

    # Results layout

    fig_i = make_currents_figure(
        t=last_result["t"],
        is_arr=last_result["is"],
        i2_arr=last_result["i2"],
        is_rms=last_result["is_rms"],
        i2_rms=last_result["i2_rms"]
    )
    st.plotly_chart(fig_i, use_container_width=True)

    fig_f = make_flux_excitation_figure(
        t=last_result["t"],
        lam=last_result["lam"],
        ie_arr=last_result["ie"]
    )
    st.plotly_chart(fig_f, use_container_width=True)
    st.markdown("### Derived parameters")
    st.json(last_derived, expanded=True)


with tab_export:
    st.subheader("Baixar resultados")

    if "last_result" not in st.session_state:
        st.info("Rode uma simulação na aba Simulação para habilitar exportação.")
        st.stop()

    res = st.session_state["last_result"]
    inp_used = st.session_state["last_inp"]
    derived_used = st.session_state["last_derived"]

    col_a, col_b = st.columns(2)

    with col_a:
        csv_bytes = results_to_csv_bytes(res=res)
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="ct_sat_results.csv",
            mime="text/csv"
        )

    with col_b:
        json_bytes = results_to_json_bytes(inp=inp_used, derived=derived_used)
        st.download_button(
            label="Download JSON (inp + derived)",
            data=json_bytes,
            file_name="ct_sat_run.json",
            mime="application/json"
        )

    st.caption("CSV inclui: t, lambda, is, ie, i2, is_rms, i2_rms")


with tab_help:
    st.subheader("Notas rápidas")

    st.markdown("### Significado dos parâmetros")
    st.write("- **Vs**: tensão RMS no secundário no ensaio de excitação quando **Ie = 10 A**.")
    st.write("- **S**: expoente da curva de saturação em log-log; controla rigidez perto da saturação.")
    st.write("- **Rw, Rb, Xb**: resistência do enrolamento e burden; **Xb** define **Lb = Xb/ω**.")
    st.write("- **Off**: offset DC em pu na corrente de falta ideal **is(t)** (limitado a [-1, 1]).")
    st.write("- **XoverR**: usado para **Tau1 = (X/R)/ω**, decaimento do componente DC.")
    st.write("- **Lamrem**: remanência (pu de Lamsat) aplicada como condição inicial em λ.")

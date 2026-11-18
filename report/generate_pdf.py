"""
PathSnap: Technical Report PDF Compiler
Generates the official consolidated PDF report for Deliverable 2:
- Formats Sections 1 through 8 strictly according to guidelines
- Formats Tables 3.1, 4.1, and 5.1
- Embeds high-resolution diagnostic charts, confusion matrices, and geographical map renders
Outputs: report/PathSnap_Technical_Report.pdf
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)

def build_pdf(output_pdf_path: str = "report/PathSnap_Technical_Report.pdf"):
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f2b48'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1d63a8'),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f2b48'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#555555'),
        alignment=1, # Center
        spaceAfter=8
    )

    story = []

    # Title Banner
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("Course UE24CS352A: Machine Learning | PES University, RR Campus", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1d63a8'), spaceAfter=10))

    # Executive Overview
    story.append(Paragraph("<b>1. Executive Summary & Problem Context</b>", h1_style))
    story.append(Paragraph(
        "GPS tracking systems in urban environments suffer from severe location inaccuracies caused by high-rise building reflections "
        "(multipath effects), sensor noise, low sampling intervals, and signal dropouts. PathSnap solves this problem by matching noisy "
        "GPS trajectories (.track) to a directed road network (.nodes, .arcs) to reconstruct realistic, continuous vehicle routes (.route). "
        "Our solution integrates KDTree spatial candidate generation, geometric/kinematic feature engineering, a comparative benchmark "
        "of five machine learning models across diverse algorithm families, Bayesian hyperparameter optimization (Optuna TPE), and a sequential "
        "Hidden Markov Model (HMM) solved via Viterbi decoding. The system is deployed via a two-laptop distributed architecture: a FastAPI "
        "server hosting the models on Laptop 1 and an evaluator dashboard on Laptop 2.",
        body_style
    ))

    # Section 2: EDA
    story.append(Paragraph("<b>2. Dataset Understanding & Exploratory Data Analysis (EDA)</b>", h1_style))
    story.append(Paragraph(
        "The complete dataset consists of 90 development trips (80 Training, 10 Student-Testing) totaling 227,577 GPS observations and "
        "34,813 traversed road segments across Germany (mean latitude ~49.85°N). All coordinates were transformed into local metric coordinates (meters) "
        "via geodesic scaling (1° lat = 111,139 m, 1° lon = 111,139 * cos(lat) m) to avoid angular distortion.",
        body_style
    ))
    story.append(Paragraph("• <b>Data Quality & Missing Values:</b> Audited all files across both partitions: 0 missing values/NaNs found, 100% chronologically monotonic timestamps.", bullet_style))
    story.append(Paragraph("• <b>Sampling Interval Analysis:</b> 96.68% of consecutive GPS observations are spaced exactly 1.0s apart (nominal 1 Hz), 1.96% at 2.0s, and 1.35% exhibit gaps > 2.0s.", bullet_style))
    story.append(Paragraph("• <b>Road Network Extent:</b> Local subnetwork graphs vary from 1,114 nodes up to 309,939 nodes, with average node degree ~2.3, characteristic of directed planar street topologies.", bullet_style))
    story.append(Paragraph("• <b>Feature Extraction:</b> 8 features engineered per candidate: perpendicular distance, heading discrepancy, start/end node distances, projection ratio, segment length, vehicle speed, and distance proximity rank.", bullet_style))

    # Add EDA Charts
    if os.path.exists("results/eda_sampling_intervals.png") and os.path.exists("results/eda_trip_lengths.png"):
        img_table = Table([
            [Image("results/eda_sampling_intervals.png", width=3.4*inch, height=2.1*inch),
             Image("results/eda_trip_lengths.png", width=3.4*inch, height=2.1*inch)]
        ], colWidths=[3.5*inch, 3.5*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)
        story.append(Paragraph("Figure 1: (Left) GPS Sampling Interval Distribution (96.7% at 1s); (Right) Trip Observation Length Distribution.", caption_style))

    # Section 3: Model Experimentation
    story.append(Paragraph("<b>3. Model Experimentation and Comparison</b>", h1_style))
    story.append(Paragraph(
        "Teams must train and compare at least five distinct models from multiple algorithm families. We evaluated 5 models using 5-Fold GroupKFold "
        "(grouped strictly by trip ID to prevent cross-trip leakage). Results are consolidated in Table 1 below:",
        body_style
    ))

    # Table 1: Model Comparison
    table_data = [
        ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Acc", "Route Cont.", "Test Macro F1", "Log Loss", "Latency"],
        ["Logistic Reg.", "Linear / Prob.", "C=1.0, L-BFGS, Scaler", "0.8157", "65.85%", "88.61%", "0.8183", "0.1692", "0.000 ms"],
        ["KNN", "Instance-Based", "k=9, Distance-Weighted", "0.8069", "64.55%", "87.02%", "0.8154", "0.5302", "0.029 ms"],
        ["Random Forest", "Ensemble", "N=100, Depth=14, MinSplit=10", "0.8291", "66.86%", "89.51%", "0.8328", "0.1491", "0.002 ms"],
        ["LightGBM", "Boosting", "N=120, lr=0.08, Leaves=31", "0.8294", "66.47%", "88.98%", "0.8330", "0.1585", "0.001 ms"],
        ["MLP Classifier", "Neural Networks", "Layers=(64,32), ReLU, Adam", "0.8247", "65.59%", "88.28%", "0.8282", "0.1518", "0.001 ms"]
    ]
    t1 = Table(table_data, colWidths=[1.1*inch, 0.95*inch, 1.4*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.55*inch, 0.65*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f2b48')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (2,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: Consolidated Model Comparison across 5 algorithm families (Section 3.1).", caption_style))

    # Compulsory Model Questions Summary
    story.append(Paragraph("<b>Compulsory Model Analysis (Section 3.2):</b>", h2_style))
    story.append(Paragraph(
        "<b>• Logistic Regression:</b> Fast linear baseline with interpretable odds ratios. Exhibits mild underfitting due to inability to model non-linear spatial splits.<br/>"
        "<b>• KNN:</b> Non-parametric distance similarity. Suffers from high log loss (0.5302) due to uncalibrated voting probabilities and elevated query latency.<br/>"
        "<b>• Random Forest:</b> Robust non-linear decision thresholds. Outstanding generalization (Top-1 66.86%, Macro F1 0.8328) with zero overfitting under max_depth=14.<br/>"
        "<b>• LightGBM:</b> Highest CV Macro F1 (0.8294) and sub-millisecond inference (0.001 ms). Selected for Bayesian optimization.<br/>"
        "<b>• MLP Classifier:</b> Smooth latent representations across coordinates and speeds. Converged in 25 epochs with early stopping.",
        body_style
    ))

    # Comparison Plots
    if os.path.exists("results/model_f1_comparison.png") and os.path.exists("results/confusion_matrix_lightgbm.png"):
        img_table2 = Table([
            [Image("results/model_f1_comparison.png", width=3.4*inch, height=2.0*inch),
             Image("results/confusion_matrix_lightgbm.png", width=2.5*inch, height=2.0*inch)]
        ], colWidths=[3.8*inch, 3.2*inch])
        img_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(img_table2)
        story.append(Paragraph("Figure 2: (Left) Cross-Validation vs Test Macro F1; (Right) LightGBM Test Confusion Matrix.", caption_style))

    story.append(PageBreak())

    # Section 4: Bayesian Optimization
    story.append(Paragraph("<b>4. Bayesian Hyperparameter Optimisation (Section 4)</b>", h1_style))
    story.append(Paragraph(
        "LightGBM was tuned using Optuna's Tree-structured Parzen Estimator (TPE) over 20 trials optimizing 3-Fold GroupKFold CV Macro F1. "
        "The search space evaluated `n_estimators` [80-200], `learning_rate` [0.02-0.20], `num_leaves` [20-64], `max_depth` [5-15], "
        "`min_child_samples` [15-80], `subsample` [0.65-0.95], and `colsample_bytree` [0.65-0.95].",
        body_style
    ))
    story.append(Paragraph("<b>Best Hyperparameters:</b> `n_estimators=160`, `learning_rate=0.0256`, `num_leaves=50`, `max_depth=7`, `min_child_samples=49`, `subsample=0.692`, `colsample=0.947`.", body_style))

    opt_table_data = [
        ["Model Variant", "CV Macro F1", "Student-Test Top-1 Acc", "Student-Test Macro F1", "Log Loss", "Inference Time"],
        ["Untuned LightGBM", "0.8294", "66.47%", "0.8330", "0.1585", "0.0010 ms"],
        ["Tuned LightGBM", "0.8337 (+0.004)", "66.84% (+0.37%)", "0.8352 (+0.002)", "0.1477 (-6.8%)", "0.0014 ms"]
    ]
    t2 = Table(opt_table_data, colWidths=[1.8*inch, 1.1*inch, 1.4*inch, 1.3*inch, 0.9*inch, 0.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f2b48')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Untuned vs Tuned Model Performance Comparison.", caption_style))

    # Section 5: HMM and Viterbi
    story.append(Paragraph("<b>5. HMM and Viterbi Route Reconstruction (Section 5)</b>", h1_style))
    story.append(Paragraph(
        "A Hidden Markov Model was implemented to enforce physical road-network connectivity over isolated pointwise predictions:<br/>"
        "• <b>Hidden States:</b> Candidate road segments ($arc\_id$) queried per GPS point.<br/>"
        "• <b>Emission Probabilities:</b> Combines spatial Gaussian sensor likelihood with classifier confidence: "
        "ln P(z|s) = -0.5*(d/σ_z)^2 + ln(P_model + 1e-4) (σ_z = 12m).<br/>"
        "• <b>Transition Probabilities:</b> Incorporates local graph reachability (1-3 hops) vs GPS displacement d_gps: "
        "ln P(v|u) = -0.4 - |d_graph - d_gps| / β (β = 25m). Disconnected transitions are penalized (-10.0).<br/>"
        "• <b>GPS Gaps:</b> Steps with Δt > 6s reset transition priors to uniform, preventing penalization across tunnels or dropouts.<br/>"
        "• <b>Viterbi Algorithm:</b> Log-space dynamic programming trellis backtracks to identify the global maximum-likelihood path.",
        body_style
    ))

    hmm_table_data = [
        ["Metric Evaluated", "Classifier-Only", "Classifier + HMM / Viterbi", "Absolute Improvement"],
        ["Pointwise Top-1 Accuracy", "75.91%", "80.72%", "+4.81% overall gain (up to 98.1%)"],
        ["Route Topological Continuity", "95.74%", "99.59%", "+3.85% (eliminates disconnected hops)"],
        ["Route Sequence Recovery F1", "0.7420", "0.8670", "+0.1250 sequence overlap gain"],
        ["Total Pipeline Latency / Trip", "48 ms", "490 ms", "Real-time sub-second execution"]
    ]
    t3 = Table(hmm_table_data, colWidths=[2.2*inch, 1.4*inch, 1.8*inch, 1.8*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f2b48')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t3)
    story.append(Paragraph("Table 3: Impact of HMM / Viterbi Sequential Decoding across all 10 Student-Test Trips.", caption_style))

    # Section 6: Geographical Visualisation
    story.append(Paragraph("<b>6. Geographical Visualisation Using Folium (Section 6)</b>", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps were generated displaying raw GPS points (blue circle markers), road network geometry (gray lines), "
        "ground-truth traversed path (solid green), and HMM reconstructed path (dashed red). Below is a representative export for Trip 00000007 "
        "(1,354 GPS points, 100.0% route continuity, 0.9719 Sequence F1):",
        body_style
    ))
    if os.path.exists("results/folium_map_static_render.png"):
        story.append(Image("results/folium_map_static_render.png", width=6.5*inch, height=3.6*inch))
        story.append(Paragraph("Figure 3: High-Resolution Folium Map Visualisation for Trip 00000007 (Blue=GPS, Green=Truth, Red=HMM Predicted).", caption_style))

    # Section 7: API and UI Deployment
    story.append(Paragraph("<b>7. API, UI and Distributed Deployment (Section 7)</b>", h1_style))
    story.append(Paragraph(
        "The system complies with the Two-Laptop Deployment Architecture:<br/>"
        "• <b>Laptop 1 (FastAPI Server):</b> Bound to 0.0.0.0:8000. Hosts trained model artifacts and server-side ground truth. "
        "Logs all requests, candidates processed, HMM decoding times, and evaluation scores live to the CMD terminal.<br/>"
        "• <b>Laptop 2 (Evaluator Dashboard):</b> Communicates with Laptop 1 via REST API. Displays the 10 evaluation trips, "
        "renders interactive comparison Folium maps, and displays accuracy/continuity metric cards within 500 ms.<br/>"
        "• <b>Zero Data Leakage:</b> Ground-truth reference routes (.route) remain strictly on Laptop 1 and are never transmitted to the UI beforehand.",
        body_style
    ))

    # Section 8: Academic Integrity
    story.append(Paragraph("<b>8. Academic Integrity & Reproducibility</b>", h1_style))
    story.append(Paragraph(
        "All code, pipelines, and models were independently authored and tested. All results in this report are verifiable via the submitted "
        "repository scripts (`python models/train_all.py`, `python hmm/evaluation.py`). Evaluator-testing data remained untouched during development.",
        body_style
    ))

    doc.build(story)
    print(f"Technical Report PDF successfully generated: {output_pdf_path}")
    return output_pdf_path

if __name__ == "__main__":
    build_pdf()

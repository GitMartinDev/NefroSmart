from flask import Blueprint, request, jsonify, send_file
from .model import predecir
from io import BytesIO
from reportlab.pdfgen import canvas

prediction_bp = Blueprint("prediction", __name__)

@prediction_bp.route("/predict", methods=["POST"])
def predict():
    try:
        datos = request.json.get("valores")
        if datos is None or not isinstance(datos, list):
            return jsonify({"error": "Debes enviar una lista de valores numéricos"}), 400

        resultado = predecir(datos)
        return jsonify({"resultado": int(resultado)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route('/generar_pdf', methods=['GET'])
def generar_pdf():
    buffer = BytesIO()
    p = canvas.Canvas(buffer)

    # Contenido del PDF de ejemplo
    p.setFont("Helvetica", 14)
    p.drawString(100, 800, "Resultado de Predicción de Insuficiencia Renal")
    p.drawString(100, 770, "Este PDF fue generado correctamente desde Flask.")
    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="resultado_prediccion.pdf",
        mimetype='application/pdf'
    )

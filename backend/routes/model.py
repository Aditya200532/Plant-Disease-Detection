from __future__ import annotations

from fastapi import APIRouter, Request
from backend.schemas.schemas import ModelInfoResponse

router = APIRouter()

DISEASE_INFO = {
    "Bacterial spot": {
        "description": "Bacterial spot is caused by Xanthomonas species. It produces small, dark, water-soaked lesions on leaves.",
        "prevention": "Use disease-free seeds, practice crop rotation, avoid overhead irrigation, and remove infected plant debris.",
    },
    "Early blight": {
        "description": "Early blight is caused by Alternaria solani. It produces dark concentric rings (target spots) on older leaves.",
        "prevention": "Use resistant varieties, practice crop rotation, maintain adequate spacing for airflow, and remove affected leaves.",
    },
    "Late blight": {
        "description": "Late blight is caused by Phytophthora infestans. It produces large, irregularly shaped water-soaked lesions.",
        "prevention": "Use resistant varieties, avoid overhead watering, ensure good air circulation, and remove infected plants promptly.",
    },
    "Leaf Mold": {
        "description": "Leaf mold is caused by Passalora fulva. It appears as pale green to yellow spots on upper leaf surfaces with olive-green mold underneath.",
        "prevention": "Improve air circulation, reduce humidity, avoid wetting leaves, and use resistant varieties.",
    },
    "Septoria leaf spot": {
        "description": "Septoria leaf spot is caused by Septoria lycopersici. It produces many small circular spots with dark borders and gray centers.",
        "prevention": "Remove infected leaves, practice crop rotation, avoid overhead irrigation, and use mulch to reduce soil splash.",
    },
    "Spider mites Two spotted spider mite": {
        "description": "Two-spotted spider mites (Tetranychus urticae) cause stippling, discoloration, and webbing on leaves.",
        "prevention": "Maintain adequate humidity, introduce natural predators, keep plants well-watered, and remove heavily infested leaves.",
    },
    "Target Spot": {
        "description": "Target spot is caused by Corynespora cassiicola. It produces brown lesions with concentric rings on leaves.",
        "prevention": "Use resistant varieties, maintain proper plant spacing, practice crop rotation, and remove infected debris.",
    },
    "Tomato YellowLeaf  Curl Virus": {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies. Leaves curl upward and become yellow.",
        "prevention": "Control whitefly populations, use resistant varieties, use reflective mulch, and remove infected plants.",
    },
    "Tomato mosaic virus": {
        "description": "Tomato Mosaic Virus (ToMV) causes mottled light and dark green patterns on leaves and may cause leaf curling.",
        "prevention": "Use resistant varieties, sanitize tools and hands, avoid tobacco products near plants, and remove infected plants.",
    },
    "Healthy": {
        "description": "The plant appears healthy with no visible signs of disease.",
        "prevention": "Continue regular care: adequate watering, proper nutrition, good air circulation, and monitoring for pests.",
    },
}


@router.get("/model-info")
async def model_info(request: Request):
    metadata = request.app.state.metadata
    metrics = request.app.state.metrics
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        number_of_classes=metadata["number_of_classes"],
        image_size=metadata["image_size"],
        class_names=metadata["class_names"],
        test_accuracy=metrics.get("test_accuracy"),
        precision_weighted=metrics.get("precision_weighted"),
        recall_weighted=metrics.get("recall_weighted"),
        f1_weighted=metrics.get("f1_weighted"),
    )


@router.get("/disease-info")
async def disease_info():
    return DISEASE_INFO


@router.get("/disease-info/{disease_name}")
async def disease_info_by_name(disease_name: str):
    info = DISEASE_INFO.get(disease_name)
    if info:
        return info
    for key, value in DISEASE_INFO.items():
        if disease_name.lower() in key.lower():
            return value
    return {
        "description": "Information not available for this disease.",
        "prevention": "Consult a local agricultural extension service for guidance.",
    }

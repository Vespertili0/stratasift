import ollama
import httpx
from typing import Dict, Any, Optional


def check_ollama_cloud_health(
    base_url: str, model_name: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Validate Ollama Cloud connection and model synchronisation.

    This function verifies connectivity to the Ollama daemon and checks
    if the requested cloud model has been synchronised to the machine.
    It catches connection and authentication errors gracefully.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = ollama.Client(host=base_url, headers=headers)

    # 1. Verify connection to the Ollama daemon
    try:
        # We invoke list() to check if the daemon is up and responding
        client.list()
    except (httpx.ConnectError, ConnectionRefusedError, ollama.RequestError):
        return {
            "success": False,
            "message": f"Could not connect to the Ollama daemon at '{base_url}'. Please ensure Ollama is running.",
            "error_type": "connection",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to connect to the Ollama daemon: {str(e)}",
            "error_type": "connection",
        }

    # 2. Check if the cloud model is synchronised
    try:
        client.show(model=model_name)
        return {
            "success": True,
            "message": f"Verified connectivity [SUCCESS] (Model: {model_name})",
        }
    except ollama.ResponseError as e:
        err_msg = str(e).lower()
        # Check if the error is due to authentication / not signed in
        is_auth_error = (
            e.status_code in (401, 403)
            or "unauthorized" in err_msg
            or "sign in" in err_msg
            or "signin" in err_msg
            or "auth" in err_msg
        )
        if is_auth_error:
            return {
                "success": False,
                "message": "Authentication required. Please run 'ollama signin' to authenticate to ollama.com.",
                "error_type": "auth",
            }
        elif e.status_code == 404 or "not found" in err_msg:
            return {
                "success": False,
                "message": f"Model '{model_name}' has not been synchronised to this machine.",
                "error_type": "model_missing",
            }
        else:
            return {
                "success": False,
                "message": f"Ollama response error ({e.status_code}): {e.error}",
                "error_type": "response",
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error during model verification: {str(e)}",
            "error_type": "unexpected",
        }

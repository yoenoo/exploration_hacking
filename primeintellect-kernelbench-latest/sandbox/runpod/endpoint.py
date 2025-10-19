import requests
import json
import os
from typing import Optional, Dict, List, Any


class RunPodEndpointManager:
    """
    A client for managing RunPod Serverless endpoints via the REST API.
    
    Documentation: https://docs.runpod.io/api-reference/endpoints
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RunPod endpoint manager.
        
        Args:
            api_key: RunPod API key. If not provided, will try to get from RUNPOD_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in RUNPOD_API_KEY environment variable")
        
        self.base_url = "https://rest.runpod.io/v1/endpoints"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request and handle errors."""
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code} error: {response.text}"
            raise Exception(error_msg) from e
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}") from e
    
    def create_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_type_ids: List[str],
        workers_max: int = 1,
        workers_min: int = 0,
        gpu_count: int = 1,
        idle_timeout: int = 5,
        execution_timeout_ms: int = 600000,
        scaler_type: str = "QUEUE_DELAY",
        scaler_value: int = 4,
        flashboot: bool = True,
        data_center_ids: Optional[List[str]] = None,
        allowed_cuda_versions: Optional[List[str]] = None,
        network_volume_id: Optional[str] = None,
        compute_type: str = "GPU",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new serverless endpoint.
        
        Args:
            name: Name for the endpoint
            template_id: ID of the template to use
            gpu_type_ids: List of GPU types (e.g., ["NVIDIA GeForce RTX 4090"])
            workers_max: Maximum number of workers
            workers_min: Minimum number of workers (always running)
            gpu_count: Number of GPUs per worker
            idle_timeout: Seconds before idle worker scales down (1-3600)
            execution_timeout_ms: Max milliseconds per request
            scaler_type: "QUEUE_DELAY" or "REQUEST_COUNT"
            scaler_value: Scaling threshold value
            flashboot: Enable flash boot
            data_center_ids: List of data center IDs
            allowed_cuda_versions: List of allowed CUDA versions
            network_volume_id: Optional network volume ID
            compute_type: "GPU" or "CPU"
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the created endpoint information
        """
        payload = {
            "name": name,
            "templateId": template_id,
            "gpuTypeIds": gpu_type_ids,
            "workersMax": workers_max,
            "workersMin": workers_min,
            "gpuCount": gpu_count,
            "idleTimeout": idle_timeout,
            "executionTimeoutMs": execution_timeout_ms,
            "scalerType": scaler_type,
            "scalerValue": scaler_value,
            "flashboot": flashboot,
            "computeType": compute_type,
        }
        
        if data_center_ids:
            payload["dataCenterIds"] = data_center_ids
        if allowed_cuda_versions:
            payload["allowedCudaVersions"] = allowed_cuda_versions
        if network_volume_id:
            payload["networkVolumeId"] = network_volume_id
        
        # Add any additional parameters
        payload.update(kwargs)
        
        print(f"Creating endpoint: {name}")
        result = self._make_request("POST", self.base_url, json=payload)
        print(f"✅ Endpoint created successfully! ID: {result.get('id')}")
        return result
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all endpoints.
        
        Returns:
            List of endpoint dictionaries
        """
        print("Fetching all endpoints...")
        result = self._make_request("GET", self.base_url)
        endpoints = result if isinstance(result, list) else result.get('endpoints', [])
        print(f"✅ Found {len(endpoints)} endpoint(s)")
        return endpoints
    
    def get_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get details for a specific endpoint.
        
        Args:
            endpoint_id: The endpoint ID
            
        Returns:
            Dict containing endpoint information
        """
        print(f"Fetching endpoint: {endpoint_id}")
        url = f"{self.base_url}/{endpoint_id}"
        result = self._make_request("GET", url)
        print(f"✅ Endpoint retrieved: {result.get('name')}")
        return result
    
    def update_endpoint(
        self,
        endpoint_id: str,
        name: Optional[str] = None,
        workers_max: Optional[int] = None,
        workers_min: Optional[int] = None,
        idle_timeout: Optional[int] = None,
        scaler_type: Optional[str] = None,
        scaler_value: Optional[int] = None,
        gpu_type_ids: Optional[List[str]] = None,
        data_center_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing endpoint.
        
        Args:
            endpoint_id: The endpoint ID to update
            name: New name for the endpoint
            workers_max: New maximum number of workers
            workers_min: New minimum number of workers
            idle_timeout: New idle timeout in seconds
            scaler_type: New scaler type
            scaler_value: New scaler value
            gpu_type_ids: New list of GPU types
            data_center_ids: New list of data center IDs
            **kwargs: Additional parameters to update
            
        Returns:
            Dict containing updated endpoint information
        """
        payload = {}
        
        if name is not None:
            payload["name"] = name
        if workers_max is not None:
            payload["workersMax"] = workers_max
        if workers_min is not None:
            payload["workersMin"] = workers_min
        if idle_timeout is not None:
            payload["idleTimeout"] = idle_timeout
        if scaler_type is not None:
            payload["scalerType"] = scaler_type
        if scaler_value is not None:
            payload["scalerValue"] = scaler_value
        if gpu_type_ids is not None:
            payload["gpuTypeIds"] = gpu_type_ids
        if data_center_ids is not None:
            payload["dataCenterIds"] = data_center_ids
        
        # Add any additional parameters
        payload.update(kwargs)
        
        print(f"Updating endpoint: {endpoint_id}")
        url = f"{self.base_url}/{endpoint_id}"
        result = self._make_request("PATCH", url, json=payload)
        print(f"✅ Endpoint updated successfully!")
        return result
    
    def delete_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Delete an endpoint.
        
        Args:
            endpoint_id: The endpoint ID to delete
            
        Returns:
            Dict containing deletion confirmation
        """
        print(f"Deleting endpoint: {endpoint_id}")
        url = f"{self.base_url}/{endpoint_id}"
        result = self._make_request("DELETE", url)
        print(f"✅ Endpoint deleted successfully!")
        return result
    
    def print_endpoints_summary(self, endpoints: Optional[List[Dict[str, Any]]] = None):
        """
        Print a formatted summary of endpoints.
        
        Args:
            endpoints: List of endpoints. If None, will fetch all endpoints.
        """
        if endpoints is None:
            endpoints = self.list_endpoints()
        
        if not endpoints:
            print("No endpoints found.")
            return
        
        print("\n" + "="*80)
        print("ENDPOINTS SUMMARY")
        print("="*80)
        
        for i, endpoint in enumerate(endpoints, 1):
            print(f"\n{i}. {endpoint.get('name', 'N/A')}")
            print(f"   ID: {endpoint.get('id', 'N/A')}")
            print(f"   Template ID: {endpoint.get('templateId', 'N/A')}")
            print(f"   Workers: {endpoint.get('workersMin', 0)} min - {endpoint.get('workersMax', 0)} max")
            print(f"   GPU Type: {', '.join(endpoint.get('gpuTypeIds', []))}")
            print(f"   GPU Count: {endpoint.get('gpuCount', 'N/A')}")
            print(f"   Scaler: {endpoint.get('scalerType', 'N/A')} (value: {endpoint.get('scalerValue', 'N/A')})")
            print(f"   Idle Timeout: {endpoint.get('idleTimeout', 'N/A')}s")
            
            workers = endpoint.get('workers', [])
            if workers:
                print(f"   Active Workers: {len(workers)}")
        
        print("\n" + "="*80)


def main():
    """Example usage of the RunPodEndpointManager class."""
    
    # Initialize the manager
    api_key = os.environ.get("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")
    template_id = os.environ.get("RUNPOD_TEMPLATE_ID", "YOUR_TEMPLATE_ID_HERE")
    
    manager = RunPodEndpointManager(api_key=api_key)
    
    # Example: List all endpoints
    print("\n" + "="*80)
    print("LISTING ALL ENDPOINTS")
    print("="*80)
    endpoints = manager.list_endpoints()
    manager.print_endpoints_summary(endpoints)
    
    # Example: Create a new endpoint (uncomment to use)
    # print("\n" + "="*80)
    # print("CREATING NEW ENDPOINT")
    # print("="*80)
    # new_endpoint = manager.create_endpoint(
    #     name="kernelbench-endpoint",
    #     template_id=template_id,
    #     gpu_type_ids=["NVIDIA GeForce RTX 4090"],
    #     workers_max=1,
    #     workers_min=0,
    #     gpu_count=1,
    #     data_center_ids=["EU-RO-1", "CA-MTL-1"],
    #     allowed_cuda_versions=["12.8"]
    # )
    # print(json.dumps(new_endpoint, indent=2))
    
    # Example: Get a specific endpoint (uncomment and add endpoint ID)
    # print("\n" + "="*80)
    # print("GETTING SPECIFIC ENDPOINT")
    # print("="*80)
    # endpoint = manager.get_endpoint("YOUR_ENDPOINT_ID")
    # print(json.dumps(endpoint, indent=2))
    
    # Example: Update an endpoint (uncomment and add endpoint ID)
    # print("\n" + "="*80)
    # print("UPDATING ENDPOINT")
    # print("="*80)
    # updated = manager.update_endpoint(
    #     "YOUR_ENDPOINT_ID",
    #     workers_max=3,
    #     name="kernelbench-endpoint-updated"
    # )
    # print(json.dumps(updated, indent=2))
    
    # Example: Delete an endpoint (uncomment and add endpoint ID)
    # print("\n" + "="*80)
    # print("DELETING ENDPOINT")
    # print("="*80)
    # result = manager.delete_endpoint("YOUR_ENDPOINT_ID")
    # print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
import requests
import os
from typing import Optional, Dict, List, Any


class RunPodEndpointManager:
    """
    A client for managing RunPod Serverless endpoints and templates via the REST API.
    
    Documentation: 
        - Endpoints: https://docs.runpod.io/api-reference/endpoints
        - Templates: https://docs.runpod.io/api-reference/templates
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
        
        self.base_url = "https://rest.runpod.io/v1"
        self.endpoints_url = f"{self.base_url}/endpoints"
        self.templates_url = f"{self.base_url}/templates"
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
    
    # ==================== Endpoint Management Methods ====================
    
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
        result = self._make_request("POST", self.endpoints_url, json=payload)
        print(f"✅ Endpoint created successfully! ID: {result.get('id')}")
        return result
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all endpoints.
        
        Returns:
            List of endpoint dictionaries
        """
        print("Fetching all endpoints...")
        result = self._make_request("GET", self.endpoints_url)
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
        url = f"{self.endpoints_url}/{endpoint_id}"
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
        url = f"{self.endpoints_url}/{endpoint_id}"
        result = self._make_request("PATCH", url, json=payload)
        print("✅ Endpoint updated successfully!")
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
        url = f"{self.endpoints_url}/{endpoint_id}"
        result = self._make_request("DELETE", url)
        print("✅ Endpoint deleted successfully!")
        return result
    
    # ==================== Template Management Methods ====================
    
    def create_template(
        self,
        name: str,
        image_name: str,
        is_serverless: bool = True,
        docker_start_cmd: Optional[List[str]] = None,
        docker_entrypoint: Optional[List[str]] = None,
        container_disk_in_gb: int = 50,
        volume_in_gb: int = 20,
        volume_mount_path: str = "/workspace",
        env: Optional[Dict[str, str]] = None,
        ports: Optional[List[str]] = None,
        is_public: bool = False,
        category: str = "NVIDIA",
        container_registry_auth_id: Optional[str] = None,
        readme: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new template for serverless workers or pods.
        
        Args:
            name: Template name (must be unique)
            image_name: Docker image name (e.g., "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
            is_serverless: True for serverless workers, False for pods
            docker_start_cmd: Override Docker CMD (e.g., ["python", "handler.py"])
            docker_entrypoint: Override Docker ENTRYPOINT
            container_disk_in_gb: Container disk space in GB (ephemeral)
            volume_in_gb: Volume disk space in GB (persistent)
            volume_mount_path: Path where volume is mounted
            env: Environment variables as dict (e.g., {"API_KEY": "value"})
            ports: Exposed ports (e.g., ["8888/http", "22/tcp"])
            is_public: Whether template is visible to other users
            category: "NVIDIA", "AMD", or "CPU"
            container_registry_auth_id: Auth ID for private registries
            readme: Markdown readme content
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the created template information
        """
        payload = {
            "name": name,
            "imageName": image_name,
            "isServerless": is_serverless,
            "containerDiskInGb": container_disk_in_gb,
            "volumeInGb": volume_in_gb,
            "volumeMountPath": volume_mount_path,
            "isPublic": is_public,
            "category": category,
            "readme": readme,
        }
        
        if docker_start_cmd is not None:
            payload["dockerStartCmd"] = docker_start_cmd
        if docker_entrypoint is not None:
            payload["dockerEntrypoint"] = docker_entrypoint
        if env:
            payload["env"] = env
        if ports:
            payload["ports"] = ports
        if container_registry_auth_id:
            payload["containerRegistryAuthId"] = container_registry_auth_id
        
        # Add any additional parameters
        payload.update(kwargs)
        
        print(f"Creating template: {name}")
        result = self._make_request("POST", self.templates_url, json=payload)
        print(f"✅ Template created successfully! ID: {result.get('id')}")
        return result
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all templates.
        
        Returns:
            List of template dictionaries
        """
        print("Fetching all templates...")
        result = self._make_request("GET", self.templates_url)
        templates = result if isinstance(result, list) else result.get('templates', [])
        print(f"✅ Found {len(templates)} template(s)")
        return templates
    
    def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get details for a specific template.
        
        Args:
            template_id: The template ID
            
        Returns:
            Dict containing template information
        """
        print(f"Fetching template: {template_id}")
        url = f"{self.templates_url}/{template_id}"
        result = self._make_request("GET", url)
        print(f"✅ Template retrieved: {result.get('name')}")
        return result
    
    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        image_name: Optional[str] = None,
        docker_start_cmd: Optional[List[str]] = None,
        docker_entrypoint: Optional[List[str]] = None,
        container_disk_in_gb: Optional[int] = None,
        volume_in_gb: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        ports: Optional[List[str]] = None,
        readme: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing template.
        
        Args:
            template_id: The template ID to update
            name: New template name
            image_name: New Docker image name
            docker_start_cmd: New Docker CMD override
            docker_entrypoint: New Docker ENTRYPOINT override
            container_disk_in_gb: New container disk size
            volume_in_gb: New volume size
            env: New environment variables
            ports: New exposed ports
            readme: New readme content
            **kwargs: Additional parameters to update
            
        Returns:
            Dict containing updated template information
        """
        payload = {}
        
        if name is not None:
            payload["name"] = name
        if image_name is not None:
            payload["imageName"] = image_name
        if docker_start_cmd is not None:
            payload["dockerStartCmd"] = docker_start_cmd
        if docker_entrypoint is not None:
            payload["dockerEntrypoint"] = docker_entrypoint
        if container_disk_in_gb is not None:
            payload["containerDiskInGb"] = container_disk_in_gb
        if volume_in_gb is not None:
            payload["volumeInGb"] = volume_in_gb
        if env is not None:
            payload["env"] = env
        if ports is not None:
            payload["ports"] = ports
        if readme is not None:
            payload["readme"] = readme
        
        # Add any additional parameters
        payload.update(kwargs)
        
        print(f"Updating template: {template_id}")
        url = f"{self.templates_url}/{template_id}"
        result = self._make_request("PATCH", url, json=payload)
        print("✅ Template updated successfully!")
        return result
    
    def delete_template(self, template_id: str) -> Dict[str, Any]:
        """
        Delete a template.
        
        Args:
            template_id: The template ID to delete
            
        Returns:
            Dict containing deletion confirmation
        """
        print(f"Deleting template: {template_id}")
        url = f"{self.templates_url}/{template_id}"
        result = self._make_request("DELETE", url)
        print("✅ Template deleted successfully!")
        return result
    
    def create_endpoint_with_template(
        self,
        endpoint_name: str,
        template_name: str,
        image_name: str,
        gpu_type_ids: List[str],
        docker_start_cmd: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        workers_max: int = 1,
        workers_min: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method to create a template and endpoint in one call.
        
        Args:
            endpoint_name: Name for the endpoint
            template_name: Name for the template
            image_name: Docker image name
            gpu_type_ids: List of GPU types
            docker_start_cmd: Docker start command
            env: Environment variables
            workers_max: Maximum workers
            workers_min: Minimum workers
            **kwargs: Additional parameters for endpoint creation
            
        Returns:
            Dict containing both template and endpoint information
        """
        # Create the template
        template = self.create_template(
            name=template_name,
            image_name=image_name,
            is_serverless=True,
            docker_start_cmd=docker_start_cmd,
            env=env
        )
        
        template_id = template.get('id')
        
        # Create the endpoint using the new template
        endpoint = self.create_endpoint(
            name=endpoint_name,
            template_id=template_id,
            gpu_type_ids=gpu_type_ids,
            workers_max=workers_max,
            workers_min=workers_min,
            **kwargs
        )
        
        return {
            "template": template,
            "endpoint": endpoint
        }
    
    # ==================== Summary Methods ====================
    
    def print_templates_summary(self, templates: Optional[List[Dict[str, Any]]] = None):
        """
        Print a formatted summary of templates.
        
        Args:
            templates: List of templates. If None, will fetch all templates.
        """
        if templates is None:
            templates = self.list_templates()
        
        if not templates:
            print("No templates found.")
            return
        
        print("\n" + "="*80)
        print("TEMPLATES SUMMARY")
        print("="*80)
        
        for i, template in enumerate(templates, 1):
            print(f"\n{i}. {template.get('name', 'N/A')}")
            print(f"   ID: {template.get('id', 'N/A')}")
            print(f"   Image: {template.get('imageName', 'N/A')}")
            print(f"   Type: {'Serverless' if template.get('isServerless') else 'Pod'}")
            print(f"   Category: {template.get('category', 'N/A')}")
            print(f"   Public: {template.get('isPublic', False)}")
            print(f"   Container Disk: {template.get('containerDiskInGb', 'N/A')} GB")
            print(f"   Volume: {template.get('volumeInGb', 'N/A')} GB")
            
            env_vars = template.get('env', {})
            if env_vars:
                print(f"   Environment Variables: {len(env_vars)} set")
            
            ports = template.get('ports', [])
            if ports:
                print(f"   Ports: {', '.join(ports)}")
        
        print("\n" + "="*80)
    
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
    # import json  # Uncomment if you want to pretty-print results with json.dumps()
    
    # Initialize the manager
    api_key = os.environ.get("RUNPOD_API_KEY")
    
    manager = RunPodEndpointManager(api_key=api_key)
    
    # Example: List all templates
    print("\n" + "="*80)
    print("LISTING ALL TEMPLATES")
    print("="*80)
    templates = manager.list_templates()
    manager.print_templates_summary(templates)
    
    # Example: List all endpoints
    print("\n" + "="*80)
    print("LISTING ALL ENDPOINTS")
    print("="*80)
    endpoints = manager.list_endpoints()
    manager.print_endpoints_summary(endpoints)
    
    # ==================== Template Examples ====================
    
    # Example: Create a new template (uncomment to use)
    # print("\n" + "="*80)
    # print("CREATING NEW TEMPLATE")
    # print("="*80)
    # new_template = manager.create_template(
    #     name="kernelbench-template",
    #     image_name="your-docker-image:tag",
    #     is_serverless=True,
    #     docker_start_cmd=["python", "-u", "handler.py"],
    #     env={
    #         "MODEL_NAME": "gpt-4",
    #         "API_KEY": "your-key"
    #     },
    #     container_disk_in_gb=50,
    #     volume_in_gb=20
    # )
    # print(json.dumps(new_template, indent=2))
    
    # Example: Get a specific template (uncomment and add template ID)
    # print("\n" + "="*80)
    # print("GETTING SPECIFIC TEMPLATE")
    # print("="*80)
    # template = manager.get_template("YOUR_TEMPLATE_ID")
    # print(json.dumps(template, indent=2))
    
    # Example: Update a template (uncomment and add template ID)
    # print("\n" + "="*80)
    # print("UPDATING TEMPLATE")
    # print("="*80)
    # updated_template = manager.update_template(
    #     "YOUR_TEMPLATE_ID",
    #     image_name="new-docker-image:tag",
    #     env={"NEW_VAR": "new_value"}
    # )
    # print(json.dumps(updated_template, indent=2))
    
    # Example: Delete a template (uncomment and add template ID)
    # print("\n" + "="*80)
    # print("DELETING TEMPLATE")
    # print("="*80)
    # result = manager.delete_template("YOUR_TEMPLATE_ID")
    # print(json.dumps(result, indent=2))
    
    # ==================== Endpoint Examples ====================
    
    # Example: Create a new endpoint (uncomment to use)
    # print("\n" + "="*80)
    # print("CREATING NEW ENDPOINT")
    # print("="*80)
    # new_endpoint = manager.create_endpoint(
    #     name="kernelbench-endpoint",
    #     template_id="YOUR_TEMPLATE_ID",
    #     gpu_type_ids=["NVIDIA GeForce RTX 4090"],
    #     workers_max=1,
    #     workers_min=0,
    #     gpu_count=1,
    #     data_center_ids=["EU-RO-1", "CA-MTL-1"],
    #     allowed_cuda_versions=["12.8"]
    # )
    # print(json.dumps(new_endpoint, indent=2))
    
    # Example: Create template and endpoint together (uncomment to use)
    # print("\n" + "="*80)
    # print("CREATING TEMPLATE AND ENDPOINT TOGETHER")
    # print("="*80)
    # result = manager.create_endpoint_with_template(
    #     endpoint_name="kernelbench-endpoint",
    #     template_name="kernelbench-template",
    #     image_name="your-docker-image:tag",
    #     gpu_type_ids=["NVIDIA GeForce RTX 4090"],
    #     docker_start_cmd=["python", "-u", "handler.py"],
    #     env={"MODEL_NAME": "gpt-4"},
    #     workers_max=1,
    #     workers_min=0,
    #     data_center_ids=["EU-RO-1", "CA-MTL-1"]
    # )
    # print(json.dumps(result, indent=2))
    
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

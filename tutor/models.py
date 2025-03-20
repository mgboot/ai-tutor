from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from tutor.config import TutorConfig

def create_kernel_with_models(config: TutorConfig) -> Kernel:
    """Create a kernel with primary and secondary models."""
    kernel = Kernel()
    
    # Register the primary model (GPT-4o)
    kernel.add_service(
        AzureChatCompletion(
            service_id=config.primary_service_id,
            deployment_name=config.primary_deployment,
            api_key=config.primary_api_key,
            endpoint=config.primary_endpoint
        )
    )
    
    # Register the secondary model (o1)
    kernel.add_service(
        AzureChatCompletion(
            service_id=config.secondary_service_id,
            deployment_name=config.secondary_deployment,
            api_key=config.secondary_api_key,
            endpoint=config.secondary_endpoint
        )
    )
    
    return kernel

# Base bucket options at full resolution (640)
base_bucket_options = [
    (416, 960),
    (448, 864),
    (480, 832),
    (512, 768),
    (544, 704),
    (576, 672),
    (608, 640),
    (640, 608),
    (672, 576),
    (704, 544),
    (768, 512),
    (832, 480),
    (864, 448),
    (960, 416),
]

# Initialize with full resolution bucket
bucket_options = {
    640: base_bucket_options,
    # Add a pre-calculated half-resolution bucket to avoid scaling issues
    320: [
        (208, 480),
        (224, 432),
        (240, 416),
        (256, 384),
        (272, 352),
        (288, 336),
        (304, 320),
        (320, 304),
        (336, 288),
        (352, 272),
        (384, 256),
        (416, 240),
        (432, 224),
        (480, 208),
    ]
}


def find_nearest_bucket(h, w, resolution=640):
    """
    Find the nearest bucket for a given image dimensions and target resolution.
    
    Args:
        h: Height of input image
        w: Width of input image
        resolution: Target resolution (default 640)
        
    Returns:
        Tuple of (height, width) for the selected bucket
    """
    # If resolution is not in the bucket_options cache, generate scaled buckets
    if resolution not in bucket_options:
        # Calculate scale factor relative to base resolution (640)
        scale_factor = resolution / 640.0
        
        # Generate scaled buckets
        scaled_buckets = []
        for (base_h, base_w) in base_bucket_options:
            scaled_h = int(base_h * scale_factor)
            scaled_w = int(base_w * scale_factor)
            scaled_buckets.append((scaled_h, scaled_w))
        
        # Cache the scaled buckets
        bucket_options[resolution] = scaled_buckets
        print(f"Created scaled buckets for resolution {resolution}")
    
    # Find the best matching bucket
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    
    return best_bucket


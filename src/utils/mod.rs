use anyhow::anyhow;
use std::collections::HashMap;

pub fn validate_net(
    source: Vec<String>,
    target: Vec<String>,
    weights: Option<Vec<f32>>,
    verbose: bool,
) -> anyhow::Result<HashMap<String, Vec<(String, f32)>>> {
    let len_source = source.len();
    let len_target = target.len();
    if (len_source != len_target) {
        return Err(anyhow!(
            "Source and target must have the same length in order to be used for network construction!"
        ));
    }

    let mut map: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    let mut current_src: String = "".to_string();
    let mut current_target_weight: HashMap<String, f32> = HashMap::new();
    for (i, src) in source.iter().enumerate() {
        if current_src.is_empty() {
            // never set a value in there
            current_src = src.clone();
        }

        if current_src != *src {
            // incase this is a different node now
            if !current_target_weight.is_empty() {
                let data: Vec<(String, f32)> = current_target_weight
                    .iter()
                    .map(|(key, value)| (key.clone(), *value))
                    .collect();
                map.insert(current_src, data);
                // cleanup
                current_target_weight.clear();
                current_src = src.clone();
            }
        }

        let src_target = target[i].clone();
        let src_target_weight = match &weights {
            Some(we) => we[i],
            None => 1f32,
        };
        current_target_weight.insert(src_target, src_target_weight);
    }

    if !current_target_weight.is_empty() {
        let data: Vec<(String, f32)> = current_target_weight
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect();
        map.insert(current_src, data);
    }

    Ok(map)
}

//! Motor MCTS (Monte Carlo Tree Search) para optimización genómica.
//! Implementado con arquitectura SoA para alto rendimiento.

pub struct MctsNode {
    pub q_value: f32,
    pub p_prior: f32,
    pub n_visits: u32,
    pub state: Vec<f32>, // En este caso, los 4 centroides
}

pub struct MctsTree {
    pub nodes: Vec<MctsNode>,
    pub children: Vec<Vec<usize>>,
    pub parents: Vec<usize>,
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::cmp::Ordering;
use crate::compute::math::calculate_genomic_mse;

#[cfg_attr(feature = "python", pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (weights, initial_centroids=None, iterations=5000, c_puct=1.41, noise_scale=0.05)))]
pub fn optimize_centroids_mcts(
    weights: Vec<f32>,
    initial_centroids: Option<Vec<f32>>,
    iterations: usize,
    c_puct: f32,
    noise_scale: f32,
) -> Vec<f32> {
    let centroids = initial_centroids.unwrap_or_else(|| vec![-0.43, -0.1, 0.1, 0.43]);
    let mut tree = MctsTree::new(centroids, 1.0);

    for _ in 0..iterations {
        let selected_node_idx = tree.select(0, c_puct);

        if tree.nodes[selected_node_idx].n_visits > 0 {
            tree.expand(selected_node_idx, 4, noise_scale);
            let last_idx = tree.nodes.len() - 1;
            let centroids = tree.nodes[last_idx].state.clone();
            let mse = calculate_genomic_mse(weights.clone(), centroids);
            let score = 1.0 / (mse + 1e-10);
            tree.backpropagate(last_idx, score);
        } else {
            let centroids = tree.nodes[selected_node_idx].state.clone();
            let mse = calculate_genomic_mse(weights.clone(), centroids);
            let score = 1.0 / (mse + 1e-10);
            tree.backpropagate(selected_node_idx, score);
        }
    }

    // Retornar el mejor estado encontrado
    let mut best_node_idx = 0;
    let mut max_q = -1.0;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if node.q_value > max_q && node.n_visits > 0 {
            max_q = node.q_value;
            best_node_idx = idx;
        }
    }
    tree.nodes[best_node_idx].state.clone()
}

impl MctsTree {
    pub fn new(initial_state: Vec<f32>, root_p_prior: f32) -> Self {
        Self {
            nodes: vec![MctsNode {
                q_value: 0.0,
                p_prior: root_p_prior,
                n_visits: 0,
                state: initial_state,
            }],
            children: vec![vec![]],
            parents: vec![0],
        }
    }

    pub fn select(&self, node_idx: usize, c_puct: f32) -> usize {
        if self.children[node_idx].is_empty() {
            return node_idx;
        }

        let total_n = self.nodes[node_idx].n_visits as f32;
        let mut best_idx = self.children[node_idx][0];
        let mut best_score = f32::MIN;

        for &child_idx in &self.children[node_idx] {
            let node = &self.nodes[child_idx];
            // PUCT Formula: Q + C * P * (sqrt(N_parent) / (1 + N_child))
            let u_score = c_puct * node.p_prior * (total_n.sqrt() / (1.0 + node.n_visits as f32));
            let score = node.q_value + u_score;

            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        self.select(best_idx, c_puct)
    }

    pub fn expand(&mut self, node_idx: usize, num_children: usize, noise_scale: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..num_children {
            let mut new_state = self.nodes[node_idx].state.clone();
            // Aplicar mutación aleatoria (imitando el script de Python)
            for val in new_state.iter_mut() {
                *val += rng.gen_range(-noise_scale..noise_scale);
            }
            new_state.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let child_idx = self.nodes.len();
            self.nodes.push(MctsNode {
                q_value: 0.0,
                p_prior: 1.0 / (num_children as f32), // Probabilidad uniforme inicial
                n_visits: 0,
                state: new_state,
            });
            self.children[node_idx].push(child_idx);
            self.children.push(vec![]);
            self.parents.push(node_idx);
        }
    }

    pub fn backpropagate(&mut self, mut node_idx: usize, score: f32) {
        loop {
            let node = &mut self.nodes[node_idx];
            node.n_visits += 1;
            // Actualización incremental del valor Q (media móvil)
            node.q_value += (score - node.q_value) / (node.n_visits as f32);

            if node_idx == 0 {
                break;
            }
            node_idx = self.parents[node_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_tree_initialization() {
        let tree = MctsTree::new(vec![1.0, 2.0], 1.0);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].state, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mcts_expansion() {
        let mut tree = MctsTree::new(vec![0.0], 1.0);
        tree.expand(0, 3, 0.1);
        assert_eq!(tree.nodes.len(), 4);
        assert_eq!(tree.children[0].len(), 3);
    }

    #[test]
    fn test_mcts_backpropagation() {
        let mut tree = MctsTree::new(vec![0.0], 1.0);
        tree.expand(0, 1, 0.1);
        tree.backpropagate(1, 10.0);
        assert_eq!(tree.nodes[1].n_visits, 1);
        assert_eq!(tree.nodes[1].q_value, 10.0);
        assert_eq!(tree.nodes[0].n_visits, 1);
        assert_eq!(tree.nodes[0].q_value, 10.0);
    }
}

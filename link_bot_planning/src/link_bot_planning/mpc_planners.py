def get_planner(PlannerClass,
                ...):
    self.fwd_model_dir = fwd_model_dir
    self.fwd_model_type = fwd_model_type
    self.classifier_model_dir = classifier_model_dir
    self.classifier_model_type = classifier_model_type
    self.fwd_model, self.model_path_info = model_utils.load_generic_model(self.fwd_model_dir, self.fwd_model_type)
    self.classifier_model = classifier_utils.load_generic_model(self.classifier_model_dir, self.classifier_model_type)
    self.viz_object = ompl_viz.VizObject()

    self.rrt = shooting_rrt.ShootingRRT(fwd_model=self.fwd_model,
                                        classifier_model=self.classifier_model,
                                        dt=self.fwd_model.dt,
                                        n_state=self.fwd_model.n_state,
                                        planner_params=self.planner_params,
                                        local_env_params=local_env_params,
                                        env_params=env_params,
                                        services=services,
                                        viz_object=self.viz_object,
                                        )

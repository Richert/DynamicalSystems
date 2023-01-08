# plotting
##########

# for k, v in zip(kernels, vars):
#
#     _, ax = plt.subplots(ncols=3, figsize=(10, 5))
#     ax[0].imshow(k, aspect=1.0, cmap='nipy_spectral')
#     ax[0].set_title('K')
#     ax[1].imshow(v, aspect=1.0, cmap='nipy_spectral')
#     ax[1].set_title('var(K)')
#     ax[2].plot(k[::-1, :][np.eye(k.shape[0]) > 0])
#     ax[2].set_title("off diag of K")
#     plt.show()

# plotting
# plt.plot(res["target"][-plot_length:], color="black", linestyle="dashed")
# plt.plot(res["prediction"][-plot_length:], color="orange")
# plt.plot((s @ weight_diff + res["readout_bias"]).values[-plot_length:], color="purple")
# plt.legend(["target", "prediction", "new"])
# plt.title(f"tau = {tau}, score = {res['train_score']}")
# plt.show()

# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
#
# # training scores
# ax = axes[0, 0]
# im = ax.imshow(scores)
# ax.set_ylabel(var)
# ax.set_yticks(np.arange(len(params)), labels=params)
# ax.set_xlabel("phi")
# ax.set_xticks(np.arange(len(phis)), labels=phis)
# ax.set_title("Training Scores")
# plt.colorbar(im, ax=ax)
#
# # average training scores vs. kernel peaks
# ax = axes[0, 1]
# k = data["K_diff"]
# ax.plot(k, color="blue")
# ax2 = ax.twinx()
# ax2.plot(np.mean(scores.values, axis=1), color="orange")
# ax.set_xlabel(var)
# ax.set_xticks(np.arange(len(params)), labels=params)
# ax.set_ylabel("diff", color="blue")
# ax2.set_ylabel("score", color="orange")
# ax.set_title("kernel diff vs. training score")
#
# # average training scores vs. kernel variance
# ax = axes[1, 0]
# k_vars = data["K_var"]
# ax.plot([np.mean(k) for k in k_vars], color="blue")
# ax2 = ax.twinx()
# ax2.plot(np.mean(scores.values, axis=1), color="orange")
# ax.set_xlabel(var)
# ax.set_xticks(np.arange(len(params)), labels=params)
# ax.set_ylabel("var", color="blue")
# ax2.set_ylabel("score", color="orange")
# ax.set_title("K variance vs. training score")
#
# # average training scores vs. kernel width
# ax = axes[1, 1]
# k = data["X_dim"]
# ax.plot(k, color="blue")
# ax2 = ax.twinx()
# ax2.plot(np.mean(scores.values, axis=1), color="orange")
# ax.set_xlabel(var)
# ax.set_xticks(np.arange(len(params)), labels=params)
# ax.set_ylabel("dims", color="blue")
# ax2.set_ylabel("score", color="orange")
# ax.set_title("dimensionality vs. training score")
#
# plt.tight_layout()
# plt.show()
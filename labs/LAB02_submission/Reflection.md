# Lab 2 Reflection

In this lab, both containers ran on your laptop. In production, the preprocessor would run in the warehouse datacenter and the inference API would run in Congo's main datacenter.

**How would the architecture and your `docker run` commands differ if these containers were actually running in separate datacenters?**

Consider:
- How would the preprocessor find the inference API?
- What about the shared volumes?
- What new challenges would arise?


## Your Reflection Below

In this lab, both containers ran on the same mchine therefore the preprocessor could reach the inference API through host.docker.internal which works for a single host. In production, the preprocessor would run in Congo's warehouse datacenter and the API runs in the main datacenter that approach wouldn't work. The preprocessor would need to reach the inference API over the public internet or private WAN link. In this case, I would be concerned about reliability first and foremost, what if the warehouse has a slow or intermittent network. 
The shared volume approach also breaks across datacenters. In this lab, both containers had access to the same logs folder because it was mounted from one machine's filesystem. In the case of two separate datacenters, there is not a shared file system. The API and preprocessor would then need a shared external storage layer or something along those lines. This would mean the classifications file would also have to live in an object storage since it can't be stored on a local folder this would also change the endpoint stats and and location of logs. 
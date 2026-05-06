# Stage 1

This project is a campus notification management system developed using a full stack approach. The main purpose of the system is to help students receive important notifications related to placements, results, and campus events in a prioritized manner. The application contains a frontend dashboard, backend APIs, reusable middleware, and a notification priority ranking system.

For the frontend, I used Next.js with TypeScript and Material UI to build a responsive and clean notification dashboard. The backend was developed using Node.js, Express.js, and TypeScript. REST APIs were created for user registration and notification handling.

In the system architecture, the frontend communicates with backend APIs through HTTP requests. The backend processes requests, applies middleware logging, and returns notification data. Notifications can later be stored in a database and processed asynchronously using queues for better scalability.

# Stage 2

For database design, the system mainly requires two entities: users and notifications. The users table stores student details such as id, name, email, and account creation timestamp. The notifications table stores notification id, user id, type of notification, message content, read status, priority score, and creation timestamp.

To improve database performance, indexing can be applied on user_id, priority_score, and created_at fields because notifications are usually fetched based on users and sorted by latest or highest priority notifications.

# Stage 3

Query optimization is important because notification systems may handle a large amount of data. Pagination can be used to avoid loading all notifications at once. Instead of returning thousands of rows, the API can return limited results using limit and offset queries. Proper indexing also reduces query execution time and avoids full table scans.

Another optimization technique is avoiding unnecessary repeated queries. Instead of making multiple database calls for related data, joins can be used to fetch required information efficiently.

# Stage 4

Caching can also improve performance significantly. Redis can be introduced to store recently accessed notifications, unread notification counts, and top priority notifications. When a request comes, the backend can first check the cache. If data exists in Redis, it can be returned immediately without querying the database again.

# Stage 5

For asynchronous processing, notification systems should avoid synchronous heavy operations because sending notifications one by one can slow the server. Message queues such as RabbitMQ or Apache Kafka can be used. When an event occurs, a job can be pushed into the queue and worker services can process notifications separately in the background.

# Stage 6

The project also includes a priority inbox system. Notifications are ranked using both notification type and recency. Placement notifications are considered highest priority, followed by result notifications and then event notifications. More recent notifications are ranked higher than older ones.

The priority score is calculated using a simple logic where notification type weight and recency score are combined. This helps important notifications appear at the top of the dashboard so students can quickly notice urgent updates.

Reusable middleware was implemented in the backend to log API request details such as request method, URL, timestamp, and IP address. This improves debugging and monitoring.

From a security perspective, the system can use token-based authentication, HTTPS communication, input validation, and environment variables for sensitive credentials.

The system is designed to scale in the future by adding load balancing, Redis caching, asynchronous queue processing, and database optimization techniques. Features such as push notifications, WebSockets, and real-time updates can also be integrated later.

Overall, this project demonstrates full stack development concepts including frontend UI design, backend API development, middleware handling, notification prioritization, and scalable system design.
